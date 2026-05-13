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
from typing import Any, Dict, List, Optional, Tuple

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

    # Per-head enable flags. Default: both heads enabled (legacy behaviour).
    # When ``enable_three_class_head`` is False the 3-class linear / CE term /
    # 3-class evaluation + aggregation paths are skipped end-to-end; mirror
    # for the binary head. At least one must remain True — see
    # :meth:`__post_init__`.
    enable_three_class_head: bool = True
    enable_binary_head: bool = True

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

    def __post_init__(self) -> None:
        if not (self.enable_three_class_head or self.enable_binary_head):
            raise ValueError(
                "GuidClassifierConfig: at least one of "
                "``enable_three_class_head`` or ``enable_binary_head`` must "
                "be True (both disabled leaves the classifier without a "
                "training signal)."
            )


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
            enable_three_class=cfg.enable_three_class_head,
            enable_binary=cfg.enable_binary_head,
        )
        # Surface the head flags on the classifier itself so downstream
        # consumers (inference, evaluation, lightning) can branch without
        # having to re-read ``cfg``.
        self.enable_three_class_head: bool = bool(cfg.enable_three_class_head)
        self.enable_binary_head: bool = bool(cfg.enable_binary_head)

        # Optional VAE submodule for the live-VAE training path. Set
        # externally by ``single_fold_trainer.train_fold`` when
        # ``vae.freeze_vae == False``. ``forward`` dispatches on its
        # presence + the raw-signal keys in the batch dict.
        self.vae: Optional[nn.Module] = None
        # Live-VAE chunk size on the segment axis; bounded to keep peak
        # memory predictable on long-N GUIDs.
        self.vae_chunk_size: int = 32

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

        Dispatches to :meth:`live_forward` when ``self.vae`` is attached
        and the batch carries raw VAE inputs (``fhr_st``); otherwise runs
        the cached path against the precomputed VAE features.

        Args:
            batch: Output of :func:`guid_sequence_collate_fn`.

        Returns:
            Dict with per-position head outputs (``logits_3 (B, N, 3)``,
            ``logit_bin (B, N)``, ``prob_3 (B, N, 3)``, ``prob_bin (B, N)``),
            transformer artefacts (``segment_tokens``, ``segment_context``,
            ``segment_te_summary``), and a copy of ``segment_mask`` so
            downstream consumers don't need the raw batch. Live-VAE mode
            additionally exposes ``vae_outputs`` for the auxiliary loss.
        """
        if self.vae is not None and "fhr_st" in batch:
            return self.live_forward(batch)
        return self._cached_forward(batch)

    def _cached_forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Cached-VAE forward: consumes precomputed h_y / mu / kld / α."""
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

    def live_forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Live-VAE forward: runs ``self.vae.encode_only`` per batch.

        Reshapes ``(B, N, T, *)`` raw signals into ``(B*N_valid, T, *)``,
        chunks along the segment axis by :attr:`vae_chunk_size`, runs
        :meth:`SeqVaeLagAttnV1.encode_only` per chunk, and z-scores the
        resulting ``mu_post`` / ``mu_prior`` with the VAE's running stats
        (matching what the precompute path bakes into the cache attrs).
        Padded segments produce zero tokens via the existing tokenizer
        masking, exactly like the cached path.

        Args:
            batch: Collated batch dict carrying ``fhr_st``, ``fhr_ph``,
                ``up_st``, ``up_ph`` plus the per-segment metadata.

        Returns:
            Same dict shape as :meth:`_cached_forward` plus a
            ``vae_outputs`` sub-dict with the raw VAE outputs at valid
            positions: ``mu_prior`` / ``logvar_prior`` / ``mu_post`` /
            ``logvar_post`` / ``kld_per_t`` (each shaped
            ``(M, T, ...)`` where ``M`` is the number of valid segments)
            and the per-step mask ``hat_w_v (M, T)`` used by the
            auxiliary KL loss.
        """
        if self.vae is None:
            raise RuntimeError(
                "live_forward called but self.vae is not attached. "
                "Set classifier.vae = SeqVaeLagAttnV1(...) before fit."
            )

        seg_mask = batch["segment_mask"]                      # (B, N) bool
        hat_w = batch["hat_w"]                                # (B, N, T)
        fhr_st = batch["fhr_st"]                              # (B, N, T, c_y_st)
        fhr_ph = batch["fhr_ph"]                              # (B, N, T, c_y_ph)
        up_ph = batch["up_ph"]                                # (B, N, T, c_up_ph)
        B, N, T = fhr_st.shape[:3]

        use_up_st = bool(getattr(self.vae, "use_up_st", True))
        if use_up_st:
            up_st = batch["up_st"]                            # (B, N, T, c_up_st)
            u_stream = torch.cat([up_st, up_ph], dim=-1)
        else:
            u_stream = up_ph

        flat_y_st = fhr_st.reshape(B * N, T, fhr_st.shape[-1])
        flat_y_ph = fhr_ph.reshape(B * N, T, fhr_ph.shape[-1])
        flat_u = u_stream.reshape(B * N, T, u_stream.shape[-1])
        flat_mask = seg_mask.reshape(B * N)
        flat_hat_w = hat_w.reshape(B * N, T)

        valid_idx = torch.nonzero(flat_mask, as_tuple=True)[0]  # (M,)
        M = int(valid_idx.numel())

        device = flat_y_st.device
        d_model_vae = int(self.cfg.d_model_vae)
        d_z = int(self.cfg.d_z)

        if M == 0:
            # Degenerate batch: every sample is padding. Reachable only if
            # ``min_samples_per_guid`` filtering is bypassed; provide zero
            # VAE features so the head still produces logits and the
            # token's segment_mask zeros them downstream. The attention
            # axis size must match the VAE's ``max_lag + 1`` so the
            # tokenizer's TE summary can read mean_alpha without a shape
            # mismatch.
            L_attn = int(getattr(self.vae, "max_lag", 0)) + 1
            h_y_full = torch.zeros(B, N, T, d_model_vae, device=device)
            mu_prior_norm = torch.zeros(B, N, T, d_z, device=device)
            mu_post_norm = torch.zeros(B, N, T, d_z, device=device)
            kld_per_t_full = torch.zeros(B, N, T, device=device)
            mean_alpha_full = torch.zeros(B, N, T, L_attn, device=device)
            vae_outputs: Dict[str, torch.Tensor] = {
                "mu_prior": torch.zeros(0, T, d_z, device=device),
                "logvar_prior": torch.zeros(0, T, d_z, device=device),
                "mu_post": torch.zeros(0, T, d_z, device=device),
                "logvar_post": torch.zeros(0, T, d_z, device=device),
                "kld_per_t": torch.zeros(0, T, device=device),
                "hat_w_v": torch.zeros(0, T, device=device),
            }
        else:
            valid_y_st = flat_y_st[valid_idx]
            valid_y_ph = flat_y_ph[valid_idx]
            valid_u = flat_u[valid_idx]
            valid_hat_w = flat_hat_w[valid_idx]                 # (M, T)

            chunk_size = max(1, int(self.vae_chunk_size))
            h_y_chunks: List[torch.Tensor] = []
            mu_prior_chunks: List[torch.Tensor] = []
            logvar_prior_chunks: List[torch.Tensor] = []
            mu_post_chunks: List[torch.Tensor] = []
            logvar_post_chunks: List[torch.Tensor] = []
            mean_alpha_chunks: List[torch.Tensor] = []
            kld_per_t_chunks: List[torch.Tensor] = []
            for c0 in range(0, M, chunk_size):
                c1 = min(c0 + chunk_size, M)
                enc = self.vae.encode_only(
                    valid_y_st[c0:c1],
                    valid_y_ph[c0:c1],
                    valid_u[c0:c1],
                    sample_z=False,
                )
                kld_btd = self.vae.kld_tensor(
                    mu_prior=enc["mu_prior"],
                    logvar_prior=enc["logvar_prior"],
                    mu_post=enc["mu_post"],
                    logvar_post=enc["logvar_post"],
                    mask_warmup=False,
                )                                              # (chunk, T, d_z)
                h_y_chunks.append(enc["target_state"])
                mu_prior_chunks.append(enc["mu_prior"])
                logvar_prior_chunks.append(enc["logvar_prior"])
                mu_post_chunks.append(enc["mu_post"])
                logvar_post_chunks.append(enc["logvar_post"])
                mean_alpha_chunks.append(enc["attn_weights"].mean(dim=-2))
                kld_per_t_chunks.append(kld_btd.sum(dim=-1))   # (chunk, T)

            h_y_v = torch.cat(h_y_chunks, dim=0)
            mu_prior_v = torch.cat(mu_prior_chunks, dim=0)
            logvar_prior_v = torch.cat(logvar_prior_chunks, dim=0)
            mu_post_v = torch.cat(mu_post_chunks, dim=0)
            logvar_post_v = torch.cat(logvar_post_chunks, dim=0)
            mean_alpha_v = torch.cat(mean_alpha_chunks, dim=0)
            kld_per_t_v = torch.cat(kld_per_t_chunks, dim=0)

            # Z-score with the VAE's running stats — matches the per-fold
            # ``(mean, std)`` that the precompute path bakes into the
            # cache attrs. ``fit_latent_stats`` must have been called by
            # the trainer before stage 1 begins.
            mu_mean = self.vae.mu_post_running_mean.to(device=device, dtype=mu_post_v.dtype)
            mu_std = (
                self.vae.mu_post_running_var.to(device=device, dtype=mu_post_v.dtype) + 1e-5
            ).sqrt()
            mu_post_norm_v = (mu_post_v - mu_mean) / mu_std
            mu_prior_norm_v = (mu_prior_v - mu_mean) / mu_std

            # Scatter back to (B*N, T, *) → (B, N, T, *) at valid rows.
            L_attn = mean_alpha_v.shape[-1]

            def _scatter(values: torch.Tensor, last_shape: Tuple[int, ...]) -> torch.Tensor:
                full = torch.zeros(
                    B * N, T, *last_shape, dtype=values.dtype, device=device
                )
                full[valid_idx] = values
                return full.reshape(B, N, T, *last_shape)

            h_y_full = _scatter(h_y_v, (d_model_vae,))
            mu_prior_norm = _scatter(mu_prior_norm_v, (d_z,))
            mu_post_norm = _scatter(mu_post_norm_v, (d_z,))

            kld_full_flat = torch.zeros(B * N, T, dtype=kld_per_t_v.dtype, device=device)
            kld_full_flat[valid_idx] = kld_per_t_v
            kld_per_t_full = kld_full_flat.reshape(B, N, T)

            alpha_full = torch.zeros(B * N, T, L_attn, dtype=mean_alpha_v.dtype, device=device)
            alpha_full[valid_idx] = mean_alpha_v
            mean_alpha_full = alpha_full.reshape(B, N, T, L_attn)

            vae_outputs = {
                "mu_prior": mu_prior_v,                       # (M, T, d_z)
                "logvar_prior": logvar_prior_v,
                "mu_post": mu_post_v,
                "logvar_post": logvar_post_v,
                "kld_per_t": kld_per_t_v,                     # (M, T)
                "hat_w_v": valid_hat_w,                       # (M, T)
            }

        tok = self.tokenizer(
            h_y=h_y_full,
            mu_prior_norm=mu_prior_norm,
            mu_post_norm=mu_post_norm,
            kld_per_t=kld_per_t_full,
            mean_alpha=mean_alpha_full,
            hat_w=hat_w,
            c_meta=batch["c_meta"],
            segment_mask=seg_mask,
        )
        segment_tokens = tok["segment_token"]
        h = self.transformer(
            x=segment_tokens,
            segment_mask=seg_mask,
            rel_bucket_idx=batch["rel_bucket_idx"],
        )
        head_out = self.outcome_head(h, seg_mask)

        return {
            **head_out,
            "segment_tokens": segment_tokens,
            "segment_context": h,
            "segment_te_summary": tok["u_TE"],
            "segment_mask": seg_mask,
            "vae_outputs": vae_outputs,
        }


__all__ = [
    "GuidOutcomeClassifier",
    "GuidClassifierConfig",
]
