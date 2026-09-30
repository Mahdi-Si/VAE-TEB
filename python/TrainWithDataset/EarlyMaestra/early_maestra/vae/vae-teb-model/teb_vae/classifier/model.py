"""Classifier network (SPEC §9): step encoder -> segment pooling -> token (+ context) -> [sequence
aggregator] -> heads. Pure torch; the Lightning wrapper lives in ``train.py``.

Batch and output keys follow the P3/P4 contract (CONTRACT.md). One option per config string:

* Step encoder (§9.2): Linear + LayerNorm + GELU + Dropout, then ``temporal`` = ``none`` |
  ``causal_conv`` (depthwise-separable, k 5, dilations 1/2/4, residual) | ``transformer`` (1 causal
  masked layer, 2 heads). Masked steps are zeroed after every stage.
* Pooling (§9.3): ``mean`` | ``mean_max`` | ``gated_attention`` (Ilse; ``u_t = [x_t || cues_t]``: the
  ``role: attention`` cues shape the weights, only ``x_t`` is pooled) | ``query`` (PMA k 1; cues enter
  the keys only) | ``conjunctive`` (MILLET, segment scope: per-step head logits pooled by the gated
  attention, exposed as ``step_<key>``). A segment with no valid step raises (excluded upstream).
* Token (§9.4): ``token_mlp([e || ctx])``. The optional covariate block ``batch["cov"]`` (P5; only read
  when ``n_cov > 0``) enters by ``fusion``: ``concat`` (with ctx), ``film`` (``z(1+γ)+β``, identity at
  init), ``token`` (sequence scope: added to the token), ``late`` (concatenated at every head input).
* Aggregator (§9.5): ``causal_transformer`` (pre-norm; causal + key-padding mask; per-head learned
  bias over log-bucketed |Δt| in minutes, saturating at ``time_bias_max_h``; relative times only),
  ``gru`` (input ``[z || log1p Δt]``, GRU-D decay ``h·exp(-relu(w)Δt)``; always causal), and
  ``attention_mil`` (gated attention with a cumulative masked softmax). No softmax row is ever empty
  (the diagonal is never masked; masked_softmax zeroes empty rows), so padding never yields NaN.
* Heads (§9.6): MLP(d -> hidden -> K_out), zero last layer, prior bias (``logit π`` / ``log π_k`` /
  CORAL cumulative logits with ordered biases ``b_k = b_1 - Σ_{j<k} softplus(δ_j)``). Each output ``X``
  comes with ``X_score``: binary logit | ``log Σ_{k≥1} p_k - log p_0`` | CORAL ``g + b_1 = logit P(Y ≥ 1)``
  (ranks like ``g``; every consumer reads ``σ(score)`` as P(adverse)).

:func:`aggregate` / :func:`running_aggregate` give the post-hoc segment-scope GUID scores (§11.2).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from teb_vae.classifier.config import LabelsCfg, ModelCfg, SequenceCfg, StepCfg

AGGREGATORS = ("max", "mean", "lse", "last", "topk_mean")


# ---- helpers -----------------------------------------------------------------------------------
def _proj(n_in: int, n_out: int, p: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(n_in, n_out), nn.LayerNorm(n_out), nn.GELU(), nn.Dropout(p))


def masked_softmax(logits: Tensor, valid: Tensor) -> Tensor:
    """Softmax over the last dim restricted to ``valid``; a row with no valid entry gives zeros."""
    return logits.masked_fill(~valid, torch.finfo(logits.dtype).min).softmax(-1) * valid


def _blocked(valid: Tensor, causal: bool) -> Tensor:
    """(B, L, L) bool, True where query i may not attend key j: invalid keys (and j > i if causal).
    The diagonal is never blocked, so no attention row is empty (padding cannot produce NaN)."""
    eye = torch.eye(valid.shape[-1], dtype=torch.bool, device=valid.device)
    blocked = ~valid[:, None, :]
    if causal:
        blocked = blocked | torch.ones_like(eye).triu(1)
    return blocked & ~eye


def n_params(module: nn.Module) -> int:
    """Trainable parameter count."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class _Block(nn.Module):
    """Pre-norm transformer layer. Hand-rolled because ``nn.TransformerEncoderLayer``'s eval fast path
    returns NaN with a per-head (B·H, L, L) float mask (torch 2.14); ``nn.MultiheadAttention`` is fine."""

    def __init__(self, d: int, heads: int, d_ff: int, p: float) -> None:
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=p, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Dropout(p), nn.Linear(d_ff, d))
        self.drop = nn.Dropout(p)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        h = self.norm1(x)
        x = x + self.drop(self.attn(h, h, h, attn_mask=mask, need_weights=False)[0])
        return x + self.drop(self.ff(self.norm2(x)))


class GatedAttention(nn.Module):
    """Ilse 2018 attention logits ``wᵀ[tanh(V u) ⊙ σ(U u)]`` (Appendix B); the softmax is the caller's."""

    def __init__(self, n_in: int, d_attn: int) -> None:
        super().__init__()
        self.V, self.U, self.w = nn.Linear(n_in, d_attn), nn.Linear(n_in, d_attn), nn.Linear(d_attn, 1)

    def forward(self, u: Tensor) -> Tensor:
        return self.w(torch.tanh(self.V(u)) * torch.sigmoid(self.U(u))).squeeze(-1)


# ---- step encoder (§9.2) -----------------------------------------------------------------------
class StepEncoder(nn.Module):
    def __init__(self, n_values: int, cfg: StepCfg) -> None:
        super().__init__()
        d, self.temporal = cfg.d, cfg.temporal
        self.proj = _proj(n_values, d, cfg.dropout)
        if self.temporal == "causal_conv":
            self.convs = nn.ModuleList(
                nn.Sequential(nn.ConstantPad1d((4 * dil, 0), 0.0), nn.Conv1d(d, d, 5, dilation=dil, groups=d),
                              nn.Conv1d(d, d, 1), nn.GELU())
                for dil in (1, 2, 4))
        elif self.temporal == "transformer":
            self.block = _Block(d, 2, 2 * d, cfg.dropout)

    def forward(self, x: Tensor, valid: Tensor) -> Tensor:
        """(M, T, C), (M, T) -> (M, T, d), zero at masked steps."""
        v = valid[..., None].to(x.dtype)
        h = self.proj(x) * v
        if self.temporal == "causal_conv":
            for conv in self.convs:
                h = (h + conv(h.transpose(1, 2)).transpose(1, 2)) * v
        elif self.temporal == "transformer":
            h = self.block(h, _blocked(valid, True).repeat_interleave(2, 0)) * v
        return h


# ---- pooling (§9.3) ----------------------------------------------------------------------------
class Pooling(nn.Module):
    def __init__(self, kind: str, d: int, n_attn: int, d_attn: int) -> None:
        super().__init__()
        self.kind = kind
        self.d_out = 2 * d if kind == "mean_max" else d
        if kind in ("gated_attention", "conjunctive"):
            self.gate = GatedAttention(d + n_attn, d_attn)
        elif kind == "query":
            self.query = nn.Parameter(torch.zeros(1, 1, d))
            self.attn = nn.MultiheadAttention(d, 2, kdim=d + n_attn, vdim=d, batch_first=True)

    def forward(self, x: Tensor, valid: Tensor, cues: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        """(M, T, d), (M, T), cues (M, T, C_a) | None -> pooled (M, d_out), step weights (M, T)."""
        if not bool(valid.any(-1).all()):
            raise ValueError("a segment with no valid step reached the model; it must be excluded "
                             "upstream (low_valid_frac, SPEC §9.3)")
        u = x if cues is None else torch.cat([x, cues], -1)
        if self.kind == "query":
            e, a = self.attn(self.query.expand(len(x), -1, -1), u, x, key_padding_mask=~valid)
            return e[:, 0], a[:, 0]
        if self.kind in ("mean", "mean_max"):
            a = valid / valid.sum(-1, keepdim=True)
        else:
            a = masked_softmax(self.gate(u), valid)
        e = (a[..., None] * x).sum(-2)
        if self.kind == "mean_max":
            e = torch.cat([e, x.masked_fill(~valid[..., None], float("-inf")).amax(-2)], -1)
        return e, a


# ---- sequence aggregator (§9.5) ----------------------------------------------------------------
class Aggregator(nn.Module):
    def __init__(self, cfg: SequenceCfg, d: int, d_attn: int) -> None:
        super().__init__()
        self.kind, self.causal, self.heads = cfg.kind, cfg.causal, cfg.heads
        if self.kind == "causal_transformer":
            self.blocks = nn.ModuleList(_Block(d, cfg.heads, cfg.d_ff, cfg.dropout) for _ in range(cfg.layers))
            self.norm = nn.LayerNorm(d)
            self.time_bias = nn.Embedding(cfg.time_bias_buckets, cfg.heads)
            nn.init.zeros_(self.time_bias.weight)
            self.max_h = cfg.time_bias_max_h
        elif self.kind == "gru":
            self.cell = nn.GRUCell(d + 1, d)
            self.decay = nn.Parameter(torch.full((d,), 0.1))  # > 0 so relu passes a gradient at init
        else:
            self.gate = GatedAttention(d, d_attn)

    def forward(self, z: Tensor, mask: Tensor, t_h: Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]:
        """Tokens (B, N, d), seg_mask (B, N), t_h (B, N) -> states (B, N, d), seq attention | None."""
        if self.kind == "attention_mil":
            n = mask.shape[1]
            valid = mask[:, None, :].expand(-1, n, -1)
            if self.causal:
                valid = valid & torch.ones(n, n, dtype=torch.bool, device=z.device).tril()
            a = masked_softmax(self.gate(z)[:, None, :].expand(-1, n, -1), valid)
            return a @ z, a
        t = t_h.masked_fill(~mask, 0.0)
        if self.kind == "gru":
            dt = torch.diff(t, dim=1, prepend=t[:, :1]).clamp(min=0.0)[..., None]
            h, states = z.new_zeros(z.shape[0], z.shape[2]), []
            for i in range(z.shape[1]):  # N <= ~60
                step = self.cell(torch.cat([z[:, i], torch.log1p(dt[:, i])], -1),
                                 h * torch.exp(-F.relu(self.decay) * dt[:, i]))
                h = torch.where(mask[:, i, None], step, h)
                states.append(h)
            return torch.stack(states, 1), None
        k = self.time_bias.num_embeddings  # log buckets of |Δt| in minutes, saturating at max_h
        gap_min = 60.0 * (t[:, :, None] - t[:, None, :]).abs()
        bucket = ((k - 1) * torch.log1p(gap_min) / math.log1p(60.0 * self.max_h)).floor().clamp(0, k - 1)
        bias = self.time_bias(bucket.long()).permute(0, 3, 1, 2)
        attn_mask = bias.masked_fill(_blocked(mask, self.causal)[:, None], float("-inf")).flatten(0, 1)
        for block in self.blocks:
            z = block(z, attn_mask)
        return self.norm(z), None


# ---- heads (§9.6) ------------------------------------------------------------------------------
class Head(nn.Module):
    """MLP(d -> hidden -> K_out): 1 logit (binary), K logits (multiclass) or K-1 CORAL cumulative logits."""

    def __init__(self, n_in: int, hidden: int, p: float, kind: str, k: int,
                 prior: Optional[Sequence[float]]) -> None:
        super().__init__()
        self.kind = kind
        pi = torch.as_tensor([1.0 / k] * k if prior is None else prior, dtype=torch.float64).clamp(1e-6, 1)
        pi = pi / pi.sum()
        self.mlp = nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Dropout(p))
        self.out = nn.Linear(hidden, k if kind == "multiclass" else 1, bias=kind != "ordinal")
        nn.init.zeros_(self.out.weight)
        if kind == "binary":
            nn.init.constant_(self.out.bias, math.log(pi[1] / pi[0]))
        elif kind == "multiclass":
            self.out.bias.data.copy_(pi.log())
        else:  # CORAL: b_k = logit P(Y > k), kept ordered via b_1 - cumsum(softplus(gaps))
            b = torch.logit(1 - pi.cumsum(0)[:-1])
            self.bias_first = nn.Parameter(b[:1].float())
            self.bias_gaps = nn.Parameter(torch.log(torch.expm1((b[:-1] - b[1:]).clamp(min=1e-4))).float())

    def biases(self) -> Tensor:
        """Ordered CORAL biases b_1 >= ... >= b_{K-1}."""
        return self.bias_first - torch.cat([self.bias_first.new_zeros(1), F.softplus(self.bias_gaps).cumsum(0)])

    def forward(self, h: Tensor) -> Tensor:
        o = self.out(self.mlp(h))
        return o + self.biases() if self.kind == "ordinal" else o

    def score(self, o: Tensor) -> Tensor:
        """Scalar alarm logit from the head's raw output (CONTRACT)."""
        if self.kind == "binary":
            return o[..., 0]
        if self.kind == "multiclass":
            return o[..., 1:].logsumexp(-1) - o[..., 0]
        return o[..., 0]  # CORAL g + b_1 = logit P(Y >= 1)


# ---- the network -------------------------------------------------------------------------------
class ClassifierNet(nn.Module):
    """Segment- or sequence-scope classifier (SPEC §9); ``forward(batch) -> dict`` per the contract.

    Args:
        n_values: value channels C; ``n_attn``: attention-cue channels C_a; ``n_ctx``: context width.
        model_cfg, labels_cfg: ``ModelCfg`` / ``LabelsCfg`` or their dicts.
        priors: ``{"main": [π_0..], "aux3": [π_0, π_1, π_2]}`` train-fold GUID-level priors (None: uniform).
        n_cov, fusion: covariate width (0 = none, ``batch["cov"]`` never read) and ``context.fusion``.
    """

    def __init__(self, *, n_values: int, n_attn: int, n_ctx: int, model_cfg: Any, labels_cfg: Any,
                 priors: Optional[Mapping[str, Sequence[float]]] = None, n_cov: int = 0,
                 fusion: str = "late") -> None:
        super().__init__()
        m, lab = ModelCfg.model_validate(model_cfg), LabelsCfg.model_validate(labels_cfg)
        if not m.sequence.causal and lab.strategy not in ("final_only", "mil"):
            raise ValueError("model.sequence.causal: false requires labels.strategy final_only|mil (§9.1)")
        if m.pooling.kind == "conjunctive" and m.scope != "segment":
            raise ValueError("model.pooling.kind: conjunctive is segment scope only (§9.3)")
        if n_cov and fusion == "token" and m.scope != "sequence":
            raise ValueError("context.fusion: token is sequence scope only (§9.4)")
        self.scope, self.n_ctx, self.n_cov, self.fusion = m.scope, n_ctx, n_cov, fusion
        seq, dt = m.scope == "sequence", m.token.d

        self.step_encoder = StepEncoder(n_values, m.step)
        self.pooling = Pooling(m.pooling.kind, m.step.d, n_attn, m.pooling.d_attn)
        self.token_mlp = _proj(self.pooling.d_out + n_ctx + (n_cov if fusion == "concat" else 0), dt,
                               m.token.dropout)
        self.film = nn.Linear(n_cov, 2 * dt) if n_cov and fusion == "film" else None
        if self.film is not None:  # identity at init
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
        self.cov_token = nn.Linear(n_cov, dt) if n_cov and fusion == "token" else None
        self.aggregator = Aggregator(m.sequence, dt, m.pooling.d_attn) if seq else None

        pri = priors or {}
        k = 3 if lab.task == "three_class" else 2

        def head(kind: str, k_: int, prior: Optional[Sequence[float]]) -> Head:
            return Head(dt + (n_cov if fusion == "late" else 0), m.head.hidden, m.head.dropout, kind, k_, prior)

        aux = lab.aux_3class_weight > 0
        self.head = head(lab.head, k, pri.get("main"))
        self.head_seg = head(lab.head, k, pri.get("main")) if seq and m.segment_head else None
        self.head_aux3 = head("multiclass", 3, pri.get("aux3")) if aux else None
        self.head_aux3_seg = head("multiclass", 3, pri.get("aux3")) if aux and seq and m.segment_head else None

    @property
    def causal(self) -> bool:
        """Every per-position output sees only its prefix (§9.1): segment scope, ``causal: true`` or ``gru``."""
        return self.aggregator is None or self.aggregator.causal or self.aggregator.kind == "gru"

    def _token(self, e: Tensor, ctx: Optional[Tensor], cov: Optional[Tensor]) -> Tensor:
        z = self.token_mlp(torch.cat([e] + ([ctx] if self.n_ctx else [])
                                     + ([cov] if self.n_cov and self.fusion == "concat" else []), -1))
        if self.film is not None:
            gamma, beta = self.film(cov).chunk(2, -1)
            z = z * (1 + gamma) + beta
        return z

    def _emit(self, out: Dict[str, Tensor], heads: List[Tuple[Optional[Head], str]], z: Tensor,
              cov: Optional[Tensor], a: Optional[Tensor] = None) -> None:
        if self.n_cov and self.fusion == "late":
            z = torch.cat([z, cov], -1)
        for head, key in heads:
            if head is None:
                continue
            o = head(z)
            if a is not None:  # conjunctive: attention-weighted per-step logits
                out[f"step_{key}"] = o
                o = (a[..., None] * o).sum(-2)
            out[key], out[f"{key}_score"] = o, head.score(o)

    def forward(self, batch: Mapping[str, Tensor]) -> Dict[str, Tensor]:
        seq = self.scope == "sequence"
        real = batch["seg_mask"] if seq else None
        pick = (lambda key: batch[key][real]) if seq else (lambda key: batch[key])
        x, valid = pick("x"), pick("step_mask")
        cues = pick("attn") if "attn" in batch else None
        ctx = pick("ctx") if self.n_ctx else None
        cov = batch["cov"] if self.n_cov else None
        h = self.step_encoder(x, valid)
        e, a = self.pooling(h, valid, cues)
        out: Dict[str, Tensor] = {}
        if not seq:
            seg_heads = [(self.head, "seg"), (self.head_aux3, "aux3")]
            if self.pooling.kind == "conjunctive":
                steps = h.shape[1]

                def per_step(v: Optional[Tensor]) -> Optional[Tensor]:
                    return None if v is None else v[:, None].expand(-1, steps, -1)

                self._emit(out, seg_heads, self._token(h, per_step(ctx), per_step(cov)), per_step(cov), a)
            else:
                self._emit(out, seg_heads, self._token(e, ctx, cov), cov)
            out["step_attn"] = a
            return out

        zr = self._token(e, ctx, None if cov is None else cov[real])
        z = zr.new_zeros(*real.shape, zr.shape[-1])
        z[real] = zr
        if self.cov_token is not None:
            z = z + self.cov_token(cov)
        step_attn = a.new_zeros(*real.shape, a.shape[-1])
        step_attn[real] = a
        out["step_attn"] = step_attn
        self._emit(out, [(self.head_seg, "seg"), (self.head_aux3_seg, "aux3_seg")], z, cov)
        states, seq_attn = self.aggregator(z, real, batch.get("t_h"))
        self._emit(out, [(self.head, "pos"), (self.head_aux3, "aux3_pos")], states, cov)
        if seq_attn is not None:
            out["seq_attn"] = seq_attn
        return out


# ---- post-hoc GUID aggregators (§11.2) ---------------------------------------------------------
def aggregate(scores: Any, kind: str, tau: float = 1.0) -> Any:
    """Aggregate segment logits along the last axis: ``max | mean | lse | last | topk_mean`` (k 3).

    ``lse`` = τ·log mean exp(s/τ). ``noisy_or`` is deliberately absent (§11.2). Torch in -> torch out;
    anything else (numpy, list) -> numpy (a scalar for 1-D input).
    """
    s = torch.as_tensor(scores)
    if kind == "max":
        r = s.amax(-1)
    elif kind == "mean":
        r = s.mean(-1)
    elif kind == "lse":
        r = tau * (torch.logsumexp(s / tau, -1) - math.log(s.shape[-1]))
    elif kind == "last":
        r = s[..., -1]
    elif kind == "topk_mean":
        r = s.topk(min(3, s.shape[-1]), -1).values.mean(-1)
    else:
        raise ValueError(f"unknown segment aggregator {kind!r}; expected one of {AGGREGATORS}")
    return r if isinstance(scores, Tensor) else r.numpy()[()]


def running_aggregate(scores: Any, kind: str, tau: float = 1.0) -> Any:
    """Online segment-scope GUID score: ``aggregate`` over segments <= n for every n (last axis)."""
    s = torch.as_tensor(scores)
    # ponytail: O(N^2) prefix loop, fine for N <= ~60 segments per GUID
    r = torch.stack([aggregate(s[..., : n + 1], kind, tau) for n in range(s.shape[-1])], -1)
    return r if isinstance(scores, Tensor) else r.numpy()
