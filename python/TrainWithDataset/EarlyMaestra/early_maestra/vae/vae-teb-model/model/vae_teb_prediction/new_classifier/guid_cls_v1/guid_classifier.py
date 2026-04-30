"""Top-level :class:`GuidOutcomeClassifier` for ``guid_cls_v1`` (causal AR).

Wraps the segment tokenizer → causal relative-time transformer →
:class:`PerPositionOutcomeHead`. The class is dataset-agnostic: it
consumes the dict produced by :func:`guid_sequence_collate_fn`
(cache-driven path). A live-VAE path is exposed via
:meth:`from_live_batch` (deferred to Phase 7).

Output shapes are per-position: ``logits_3 (B, N, 3)``, ``logit_bin (B, N)``.
Position ``n`` carries the model's GUID-level prediction *given history
up to position ``n``*. The training loss is a per-position masked-mean
CE/BCE with two-level reduction (per-GUID then batch).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from model.vae_teb_prediction.new_classifier.guid_cls_v1.heads import (
    PerPositionOutcomeHead,
)
from model.vae_teb_prediction.new_classifier.guid_cls_v1.segment_tokenizer import (
    VaeSegmentTokenizer,
)
from model.vae_teb_prediction.new_classifier.guid_cls_v1.temporal_transformer import (
    RelativeTimeTransformer,
)


@dataclass
class GuidClassifierConfig:
    """Resolved hyperparameters for :class:`GuidOutcomeClassifier`.

    Defaults match PRD §13. ``c_meta_dim`` is fixed by the dataset (5) and
    ``te_summary_dim`` is fixed by the tokenizer (6).
    """

    # Dimensions (auto-detected from the cache attrs at instantiation time
    # but always settable explicitly).
    d_model_vae: int = 128
    d_z: int = 24
    d_seg: int = 192
    d_model: int = 256

    # Transformer
    n_layers: int = 3
    n_heads: int = 4
    d_head: int = 64
    d_ff: int = 512
    n_rel_buckets: int = 32

    # Heads
    num_classes_multi: int = 3
    head_hidden_dim: Optional[int] = None  # defaults to ``d_model`` when None

    # Causal autoregressive flag — exposed for ablations only. Default True.
    causal: bool = True

    # Tokenizer
    c_meta_dim: int = 5
    te_summary_dim: int = 6
    late_window_steps: int = 75

    # Regularisation
    dropout: float = 0.1

    # Convenience: list of per-key overrides set by the YAML loader. Not used
    # by the model directly but lets callers serialise / round-trip the
    # effective config.
    extra: Dict[str, Any] = field(default_factory=dict)


class GuidOutcomeClassifier(nn.Module):
    """Two-level causal-autoregressive GUID-outcome classifier.

    Level 1 (per-segment): :class:`VaeSegmentTokenizer` → 256-d token.
    Level 2 (cross-segment): :class:`RelativeTimeTransformer` (causal) →
    256-d context per position.
    Head: :class:`PerPositionOutcomeHead` applied at every position.

    Args:
        cfg: Resolved hyperparameter bundle.

    Notes:
        ``forward`` accepts the dict from :func:`guid_sequence_collate_fn`.
        It never reads ``epoch`` — leakage compliance is enforced by the
        ``test_classifier_no_epoch`` unit test.
    """

    def __init__(self, cfg: GuidClassifierConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.tokenizer = VaeSegmentTokenizer(
            d_model_vae=cfg.d_model_vae,
            d_z=cfg.d_z,
            d_seg=cfg.d_seg,
            d_model=cfg.d_model,
            c_meta_dim=cfg.c_meta_dim,
            te_summary_dim=cfg.te_summary_dim,
            dropout=cfg.dropout,
            late_window_steps=cfg.late_window_steps,
        )
        self.transformer = RelativeTimeTransformer(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            d_head=cfg.d_head,
            n_layers=cfg.n_layers,
            d_ff=cfg.d_ff,
            n_buckets=cfg.n_rel_buckets,
            dropout=cfg.dropout,
            causal=cfg.causal,
        )
        self.outcome_head = PerPositionOutcomeHead(
            d_model=cfg.d_model,
            num_classes_multi=cfg.num_classes_multi,
            hidden_dim=cfg.head_hidden_dim,
            dropout=cfg.dropout,
        )

        # Marker consumed by the Lightning wrapper to bypass torch.compile.
        self.no_compile: bool = True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_cache_attrs(
        cls, cache_attrs: Dict[str, Any], cfg_overrides: Optional[Dict[str, Any]] = None
    ) -> "GuidOutcomeClassifier":
        """Construct a classifier whose dimensions match a cache file.

        Args:
            cache_attrs: ``GuidSequenceDataset.attrs`` dict.
            cfg_overrides: Optional config overrides (e.g. ``d_model``,
                ``n_layers``).

        Returns:
            A :class:`GuidOutcomeClassifier` whose ``d_model_vae`` and
            ``d_z`` match the cache.
        """
        cfg_kwargs: Dict[str, Any] = {
            "d_model_vae": int(cache_attrs["d_model"]),
            "d_z": int(cache_attrs["d_z"]),
        }
        if cfg_overrides:
            cfg_kwargs.update(cfg_overrides)
        cfg = GuidClassifierConfig(**cfg_kwargs)
        return cls(cfg)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Run the full classifier on a collated batch.

        Args:
            batch: Output of :func:`guid_sequence_collate_fn`.

        Returns:
            Dict with per-position head outputs (``logits_3 (B, N, 3)``,
            ``logit_bin (B, N)``, ``prob_3 (B, N, 3)``, ``prob_bin (B, N)``),
            transformer artefacts (``segment_tokens``, ``segment_context``,
            ``segment_te_summary``), and a copy of ``segment_mask`` so
            downstream consumers don't need the raw batch.
        """
        tok = self.tokenizer(
            h_y=batch["h_y"],
            mu_prior_norm=batch["mu_prior_norm"],
            mu_post_norm=batch["mu_post_norm"],
            kld_per_t=batch["kld_per_t"],
            mean_alpha=batch["mean_alpha"],
            hat_w=batch["hat_w"],
            c_meta=batch["c_meta"],
            segment_mask=batch["segment_mask"],
        )
        segment_tokens = tok["segment_token"]                 # (B, N, d_model)

        h = self.transformer(
            x=segment_tokens,
            segment_mask=batch["segment_mask"],
            rel_bucket_idx=batch["rel_bucket_idx"],
        )                                                     # (B, N, d_model)

        head_out = self.outcome_head(h, batch["segment_mask"])

        return {
            **head_out,
            "segment_tokens": segment_tokens,
            "segment_context": h,
            "segment_te_summary": tok["u_TE"],
            "segment_mask": batch["segment_mask"],
        }


__all__ = [
    "GuidOutcomeClassifier",
    "GuidClassifierConfig",
]
