r"""D1: one deterministic per-anchor dump of the epoch-424 s6 checkpoint over all three shards.

The pass runs the evaluation geometry: the ``val``/``test`` stages decode the dense anchor range
(stride $S = 1$, phase $\varphi = 0$), so every segment yields the anchors
$t \in [F, T_{\mathrm{ceil}}) = [134, 270)$ on the trimmed 300-step grid. Every row of the dump is
one such anchor.

**Mean decoding.** Beside the forward's own sampled branches (one common draw $\epsilon$, seeded per
batch), both branches are re-decoded at the latent means,

$$\hat y^{\,\mathrm{base}} = D(\mu^p_{t}),\qquad \hat y^{\,\mathrm{full}} = D(\mu^q_{t}),$$

through ``model.decoder(mu.gather(1, idx), persistence=out['persistence'])``, as the forward does.
The source gain of an anchor is $G = \mathrm{NLL}_{\mathrm{base}} - \mathrm{NLL}_{\mathrm{full}}$,
both scored by :func:`~teb_vae.lag_attn_rws.nets.losses.raw_sample_score` under the model's own
likelihood terms (AR(1) residual $\phi_c$, scored-cell mask, forecast step mask), summed over the
scored cells. No channel or horizon weight enters, so the score is a log-density in nats.

**Timeline.** The loader trims 15 steps (240 raw samples at 4 Hz) from each end of the stored
330-step segment. Model step $t$ is stored step $t + 15$; its 4-s window holds the trimmed raw
samples $16t - 15, \dots, 16t$ (the loader's step rule), and its time relative to delivery is
$\mathrm{epoch} + 60 + 4t$ seconds. The lag-$\ell$ key of anchor $t$ reads step $t - \ell$, so the
38-key window covers the 608 raw samples $16(t-37) - 15, \dots, 16t$ (152 s).

**Outputs** (``out/dump``):

* ``anchors.parquet``: ids and every per-anchor scalar.
* ``arrays/<name>.npy``: per-anchor vectors, row-aligned with the parquet.
* ``cells/<name>.npy``: per-cell decoder arrays $(N_{\mathrm{cell}}, 30, 76)$ for the anchors with
  $(t - F) \bmod k = 0$ ($k$ = ``cell_every``); the parquet column ``cell_row`` points into them
  ($-1$ elsewhere).
* ``schema.json``: every column and array with its shape, dtype and meaning, plus the model
  constants ($\phi_c$, the cell mask, the kept channel indices).

The dump covers every recording and stores its split code. Analyses must keep codes 0 and 2 only.

Edit :data:`RUN_ARGS` and run this file from the IDE, or pass the options on the command line.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd
import torch

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from teb_vae.lag_attn_rws.nets.losses import ar_innovation, raw_sample_score  # noqa: E402
from teb_vae.lag_attn_rws.nets.raw_masks import kl_mask  # noqa: E402
from teb_vae.neural_surgery_trf_cfs import common  # noqa: E402
from teb_vae.neural_surgery_trf_cfs.session7 import s7common as S  # noqa: E402

#: Run-button inputs.
RUN_ARGS: Dict[str, Any] = {
    "device": "cuda:0",
    "out_dir": str(S.OUT / "dump"),
    "cell_every": 5,
    "max_batches": None,
    "seed": 42,
}

#: Raw samples the loader trims from each end of a stored segment (1 min at 4 Hz).
TRIM_RAW = 240
#: Number of lag keys and the raw width of the full key window.
N_LAGS = 38
WINDOW_RAW = N_LAGS * S.RAW_PER_STEP
#: Log-variance at or below which a decoder cell counts as sitting on the floor of the clamp $(-6, 3)$.
FLOOR_LOGVAR = -5.99
#: Number of kept ``fhr_st`` channels, which come first in the target block.
N_ST = 32
#: Bit names of ``up_quality`` in bit order.
BIT_NAMES = list(S.UP_BITS)
BIT_VALUES = np.array([S.UP_BITS[name] for name in BIT_NAMES], dtype=np.int64)


# --------------------------------------------------------------------------------------------
# Shard access
# --------------------------------------------------------------------------------------------
def shard_lookup(handles: Dict[str, h5py.File]) -> Dict[Tuple[str, int], Tuple[str, int]]:
    """Index every stored segment by ``(guid, epoch)``.

    Args:
        handles: Open shard files by cohort.

    Returns:
        ``{(guid, round(epoch)): (cohort, row)}``.
    """
    table: Dict[Tuple[str, int], Tuple[str, int]] = {}
    for cohort, handle in handles.items():
        guids = [S.guid_text(g) for g in handle["guid"][:]]
        epochs = np.rint(handle["epoch"][:]).astype(np.int64)
        for row, (guid, epoch) in enumerate(zip(guids, epochs)):
            table[(guid, int(epoch))] = (cohort, row)
    return table


def run_stats(bad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Start and end (exclusive) of every run of ``True`` in a 1-D boolean array.

    Args:
        bad: Boolean array.

    Returns:
        ``(starts, ends)`` as integer arrays.
    """
    edges = np.diff(np.concatenate([[0], bad.astype(np.int8), [0]]))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def raw_validity_rows(quality: np.ndarray, anchors: np.ndarray) -> Dict[str, np.ndarray]:
    r"""Raw-level UA validity around each anchor of one segment.

    With $b_r = \mathbb 1[(q_r \,\&\, 2299) \ne 0]$ on the trimmed raw grid, anchor $t$ reads

    * ``raw_inv_60s``: $\overline{b}$ over samples $16t - 239, \dots, 16t$;
    * ``raw_inv_win``: $\overline{b}$ over the 608-sample key window $16(t-37) - 15, \dots, 16t$;
    * ``raw_inv_s_win``: invalid seconds in that window; ``raw_longest_run_s`` the longest invalid
      run inside it (clipped to the window); ``raw_n_runs`` the number of runs touching it;
    * ``anchor_step_clean``: no invalid sample among $16t - 15, \dots, 16t$;
    * ``anchor_in_invalid_run``: $b_{16t} = 1$;
    * ``dominant_bit``: the ``up_quality`` bit set on the most invalid samples of the window
      (0 when the window is clean).

    Args:
        quality: Trimmed ``up_quality`` of the segment, $(4800,)$.
        anchors: Anchor steps of the segment.

    Returns:
        One array per field, each of length ``len(anchors)``.
    """
    q = quality.astype(np.int64)
    bad = (q & S.UP_INVALID_BITS) != 0
    cs = np.concatenate([[0], np.cumsum(bad)])
    bits = (q[None, :] & BIT_VALUES[:, None]) != 0
    bits &= bad[None, :]
    cs_bits = np.concatenate([np.zeros((len(BIT_VALUES), 1), dtype=np.int64), np.cumsum(bits, axis=1)], axis=1)
    end = S.RAW_PER_STEP * anchors.astype(np.int64)  # inclusive last sample of the anchor step
    start_win = end - WINDOW_RAW + 1
    start_60 = end - 60 * S.FS + 1
    start_step = end - S.RAW_PER_STEP + 1
    n_win = cs[end + 1] - cs[start_win]
    counts = cs_bits[:, end + 1] - cs_bits[:, start_win]  # (n_bits, A)
    dominant = np.where(n_win > 0, BIT_VALUES[np.argmax(counts, axis=0)], 0)
    starts, ends = run_stats(bad)
    longest = np.zeros(len(anchors))
    n_runs = np.zeros(len(anchors), dtype=np.int16)
    for i, (lo, hi) in enumerate(zip(start_win, end + 1)):
        touch = (starts < hi) & (ends > lo)
        if touch.any():
            clipped = np.minimum(ends[touch], hi) - np.maximum(starts[touch], lo)
            longest[i] = clipped.max() / S.FS
            n_runs[i] = int(touch.sum())
    return {
        "raw_inv_60s": (cs[end + 1] - cs[start_60]) / float(60 * S.FS),
        "raw_inv_win": n_win / float(WINDOW_RAW),
        "raw_inv_s_win": n_win / float(S.FS),
        "raw_longest_run_s": longest,
        "raw_n_runs": n_runs,
        "anchor_step_clean": (cs[end + 1] - cs[start_step]) == 0,
        "anchor_in_invalid_run": bad[end],
        "dominant_bit": dominant.astype(np.int16),
    }


# --------------------------------------------------------------------------------------------
# Model-side readouts
# --------------------------------------------------------------------------------------------
class AdapterTap:
    r"""Forward hooks that keep the first call of an adapter per forward.

    The source adapter runs twice per forward: once on the recording (the key/value encode) and
    once on the null stream of the prior clock. Only the first call is the recording's, so the tap
    keeps the first input of the adapter's ``norm`` (the pre-normalisation embedding) and the first
    output of the adapter itself.
    """

    def __init__(self, adapter: torch.nn.Module) -> None:
        """Attach the hooks.

        Args:
            adapter: An :class:`AvailabilityInputAdapter`.
        """
        self.pre: List[torch.Tensor] = []
        self.out: List[torch.Tensor] = []
        self.handles = [
            adapter.norm.register_forward_pre_hook(lambda mod, inp: self.pre.append(inp[0].detach())),
            adapter.register_forward_hook(lambda mod, inp, out: self.out.append(out.detach())),
        ]

    def reset(self) -> None:
        """Forget the tensors of the previous forward."""
        self.pre.clear()
        self.out.clear()

    def remove(self) -> None:
        """Detach the hooks."""
        for handle in self.handles:
            handle.remove()


def gather_steps(x: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """Gather a per-step tensor at the anchors.

    Args:
        x: ``(B, T, ...)``.
        anchors: ``(B, A)`` long.

    Returns:
        ``(B, A, ...)``.
    """
    batch, n_anchor = anchors.shape
    index = anchors.reshape(batch, n_anchor, *([1] * (x.dim() - 2))).expand(batch, n_anchor, *x.shape[2:])
    return x.gather(1, index)


def rms(x: torch.Tensor) -> torch.Tensor:
    r"""Root mean square over the last axis, $\sqrt{\overline{x^2}}$.

    Args:
        x: Any tensor.

    Returns:
        The tensor without its last axis.
    """
    return x.float().pow(2).mean(-1).sqrt()


def score_branch(mu: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor, step_mask: torch.Tensor,
                 lk: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    r"""Per-anchor scores of one decoded branch under the model's own likelihood.

    With the scored cells $M_{a,\tau,c} = m_{a,\tau}\,c_{\tau,c}$ (step mask times cell mask), the
    per-anchor NLL is $\sum_{\tau,c} M\,\ell$ with $\ell$ from :func:`raw_sample_score`. The block
    split is ``st`` = the first 32 channels, ``ph`` = the other 44. The innovation $e$ and its
    log-variance $\log V$ come from :func:`ar_innovation`, so $z^2 = e^2 / V$.

    Args:
        mu: Forecast mean $(B, A, H, C)$.
        logvar: Decoder log-variance $(B, A, H, C)$.
        target: Scored target $(B, A, H, C)$.
        step_mask: Forecast step mask $(B, A, H)$.
        lk: ``model.forecast_likelihood_kwargs()``.

    Returns:
        Per-anchor tensors $(B, A)$: ``nll``, ``nll_st``, ``nll_ph``, ``mse_st``, ``mse_ph``,
        ``z2_st``, ``z2_ph``, ``floor_st``, ``floor_ph`` (shares of scored cells), ``n_st``,
        ``n_ph`` (scored cell counts).
    """
    cell = lk["cell_mask"]
    mask = step_mask[..., None] * cell
    nll = raw_sample_score(mu, target, likelihood="gaussian_nll", logvar=logvar, cell_mask=cell,
                           ar_coef=lk["ar_coef"], step_mask=step_mask) * mask
    innovation, logvar_eff = ar_innovation(target - mu, lk["ar_coef"], step_mask, logvar)
    z2 = innovation.pow(2) * torch.exp(-logvar_eff)
    sq = (target - mu).pow(2)
    floor = (logvar <= FLOOR_LOGVAR).to(mask.dtype)
    out: Dict[str, torch.Tensor] = {"nll": nll.sum((-1, -2))}
    for name, block in (("st", slice(0, N_ST)), ("ph", slice(N_ST, None))):
        m = mask[..., block]
        n = m.sum((-1, -2))
        den = n.clamp_min(1.0)
        out[f"nll_{name}"] = nll[..., block].sum((-1, -2))
        out[f"mse_{name}"] = (sq[..., block] * m).sum((-1, -2)) / den
        out[f"z2_{name}"] = (z2[..., block] * m).sum((-1, -2)) / den
        out[f"floor_{name}"] = (floor[..., block] * m).sum((-1, -2)) / den
        out[f"n_{name}"] = n
    return out


# --------------------------------------------------------------------------------------------
# Output buffers
# --------------------------------------------------------------------------------------------
#: Per-anchor arrays: name -> (dtype, trailing shape, meaning).
ARRAYS: Dict[str, Tuple[str, Tuple[int, ...], str]] = {
    "key_mask": ("bool", (N_LAGS,), "strict key mask K (True = usable), lag order l = step t-l"),
    "quality_anchor": ("bool", (42,), "source_quality flags v_{t,c} at the anchor step (True = valid)"),
    "quality_inv_share": ("float16", (42,), "per-channel share of invalid flags over steps t-37..t"),
    "attn": ("float16", (4, N_LAGS), "lag-attention weights alpha_{t,m,l}"),
    "invalid_mass": ("float16", (4,), "attention mass on invalid keys per head"),
    "mu_prior": ("float16", (64,), "prior mean mu^p_t"),
    "logvar_prior": ("float16", (64,), "prior log-variance"),
    "mu_post": ("float16", (64,), "posterior mean mu^q_t"),
    "logvar_post": ("float16", (64,), "posterior log-variance"),
    "kl_dim": ("float32", (64,), "closed-form KL(q||p) per latent dimension (model.kld_tensor)"),
    "kl_head": ("float32", (4,), "KL per head (16 dims each)"),
    "kl_lag": ("float32", (N_LAGS,), "source_kl_lag_map row: sum_m KL_m alpha_{m,l}"),
}

#: Per-cell arrays: name -> (dtype, meaning). Shape (N_cell, 30, 76) unless noted.
CELLS: Dict[str, Tuple[str, str]] = {
    "mu_full": ("float16", "forecast mean, full branch decoded at z = mu^q"),
    "lv_full": ("float16", "decoder log-variance, full branch at z = mu^q"),
    "mu_base": ("float16", "forecast mean, base branch decoded at z = mu^p"),
    "lv_base": ("float16", "decoder log-variance, base branch at z = mu^p"),
    "mu_z0": ("float16", "forecast mean decoded at z = 0 (persistence/skip path only)"),
    "mu_zbar": ("float16", "forecast mean decoded at z = batch mean of mu^p over its anchors"),
    "target": ("float16", "scored target"),
    "step_mask": ("bool", "forecast step mask (N_cell, 30)"),
}


def build_parser() -> argparse.ArgumentParser:
    """Command-line options; every default comes from :data:`RUN_ARGS`.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default=RUN_ARGS["device"])
    parser.add_argument("--out-dir", default=RUN_ARGS["out_dir"])
    parser.add_argument("--cell-every", type=int, default=RUN_ARGS["cell_every"])
    parser.add_argument("--max-batches", type=int, default=RUN_ARGS["max_batches"])
    parser.add_argument("--seed", type=int, default=RUN_ARGS["seed"])
    return parser


@torch.no_grad()
def source_change_check(task: Any, model: Any, batch: Dict[str, Any], lk: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Check that the mean-decoded base branch does not move when the source changes.

    The source stream and its validity mask are rolled by one sample along the batch, so every
    segment receives another segment's UA. The base branch reads only the target, so its mean
    decode and its NLL must stay identical.

    Args:
        task: The loaded task.
        model: The network.
        batch: One batch on the device.
        lk: ``model.forecast_likelihood_kwargs()``.

    Returns:
        Maximum absolute differences of the base mean, the base NLL and the full NLL.
    """
    inputs, out, target, step_mask, anchors, _ = common.forward_parts(task, model, batch)
    rolled = list(inputs)
    rolled[2] = inputs[2].roll(1, dims=0)
    rolled[5] = inputs[5].roll(1, dims=0)
    out2 = model(*rolled)
    index = anchors[:, :, None].expand(-1, -1, model.d_z)
    res = {}
    decoded = []
    for o in (out, out2):
        mu_b, lv_b = model.decoder(o["mu_prior"].gather(1, index), persistence=o["persistence"])
        mu_f, lv_f = model.decoder(o["mu_post"].gather(1, index), persistence=o["persistence"])
        decoded.append((mu_b, score_branch(mu_b, lv_b, target, step_mask, lk)["nll"],
                        score_branch(mu_f, lv_f, target, step_mask, lk)["nll"]))
    res["base_mean_maxabs"] = float((decoded[0][0] - decoded[1][0]).abs().max())
    res["nll_base_maxabs"] = float((decoded[0][1] - decoded[1][1]).abs().max())
    res["nll_full_maxabs"] = float((decoded[0][2] - decoded[1][2]).abs().max())
    return res


@torch.no_grad()
def main(args: Dict[str, Any]) -> None:
    """Run the dump.

    Args:
        args: The run arguments; see :data:`RUN_ARGS`.
    """
    out_dir = Path(args["out_dir"])
    (out_dir / "arrays").mkdir(parents=True, exist_ok=True)
    (out_dir / "cells").mkdir(parents=True, exist_ok=True)
    dev, cfg, task = S.load_model(device=args["device"])
    model = task.orig_model.eval()
    lk = model.forecast_likelihood_kwargs()
    floor_step, ceiling = int(model.warmup_period), int(model.anchor_ceiling)
    every = int(args["cell_every"])
    n_seg_total = len(common.GraphDataModule(cfg).test_dataloader().dataset)
    if args["max_batches"] is not None:
        n_seg_total = min(n_seg_total, int(args["max_batches"]) * int(cfg["general_config"]["batch_size"]["test"]))
    sel_pos = np.arange(0, ceiling - floor_step, every)
    n_cell_total = n_seg_total * len(sel_pos)
    H, C = int(model.horizon), int(lk["cell_mask"].shape[1])
    cells = {}
    for name, (dtype, _) in CELLS.items():
        shape = (n_cell_total, H) if name == "step_mask" else (n_cell_total, H, C)
        cells[name] = np.lib.format.open_memmap(out_dir / "cells" / f"{name}.npy", mode="w+",
                                                dtype=np.dtype(dtype), shape=shape)

    handles = {cohort: h5py.File(path, "r", libver="latest", swmr=True) for cohort, path in S.SHARDS.items()}
    lookup = shard_lookup(handles)
    src_tap, tgt_tap = AdapterTap(model.source_adapter), AdapterTap(model.target_adapter)

    cols: Dict[str, List[np.ndarray]] = {}
    arrs: Dict[str, List[np.ndarray]] = {name: [] for name in ARRAYS}
    checks: Dict[str, Any] = {"kl_dim_sum_maxabs": 0.0, "kl_lag_sum_maxabs": 0.0, "gate_recompute_maxabs": 0.0,
                              "source_adapter_first_call_is_h_u": True}
    cell_cursor = 0
    t0 = time.time()

    def add(name: str, value: Any) -> None:
        cols.setdefault(name, []).append(np.asarray(value))

    for bi, batch in enumerate(S.iter_batches(task, cfg, dev, limit=args["max_batches"])):
        torch.manual_seed(int(args["seed"]) + bi)
        src_tap.reset()
        tgt_tap.reset()
        inputs, out, target, step_mask, anchors, anchor_valid = common.forward_parts(task, model, batch)
        if bi == 0:
            checks["source_change"] = source_change_check(task, model, batch, lk)
            torch.manual_seed(int(args["seed"]) + bi)
            src_tap.reset()
            tgt_tap.reset()
            inputs, out, target, step_mask, anchors, anchor_valid = common.forward_parts(task, model, batch)
        B, A = anchors.shape
        h_u = out["source_state"]
        if not torch.allclose(src_tap.out[0], h_u):
            checks["source_adapter_first_call_is_h_u"] = False

        # Mean decodes of both branches.
        gidx = anchors[:, :, None].expand(-1, -1, model.d_z)
        mu_prior_a = out["mu_prior"].gather(1, gidx)
        mu_post_a = out["mu_post"].gather(1, gidx)
        mu_b, lv_b = model.decoder(mu_prior_a, persistence=out["persistence"])
        mu_f, lv_f = model.decoder(mu_post_a, persistence=out["persistence"])
        sc_b = score_branch(mu_b, lv_b, target, step_mask, lk)
        sc_f = score_branch(mu_f, lv_f, target, step_mask, lk)
        sc_bs = score_branch(out["mu_base"], out["logvar_base"], target, step_mask, lk)
        sc_fs = score_branch(out["mu_full"], out["logvar_full"], target, step_mask, lk)

        # KL decomposition and its sanity checks.
        kl = model.kld_tensor(mu_prior=out["mu_prior"], logvar_prior=out["logvar_prior"],
                              mu_post=out["mu_post"], logvar_post=out["logvar_post"])
        checks["kl_dim_sum_maxabs"] = max(checks["kl_dim_sum_maxabs"],
                                          float((kl.sum(-1) - out["kld_per_t"]).abs().max()))
        checks["kl_lag_sum_maxabs"] = max(checks["kl_lag_sum_maxabs"],
                                          float((out["source_kl_lag_map"].sum(-1) - out["kld_per_t"]).abs().max()))

        # Gate logit and its two terms.
        logit_heads = model.posterior_head.event_gate_logits(out["attended_source_heads"])
        logit_inv = model.source_invalid_gate(out["source_invalid_mass"])
        gate = out["event_gate"]
        checks["gate_recompute_maxabs"] = max(checks["gate_recompute_maxabs"],
                                              float((torch.sigmoid(logit_heads + logit_inv) - gate).abs().max()))

        # KL support, as the task's diagnostics build it.
        forecast, _ = common.forecast_mask(model.scored_weight(task._target_quality_weight(batch, task._build_raw_target(batch)[1])),
                                           model.geometry, coverage_floor=model.coverage_floor,
                                           anchors=anchors, anchor_valid=anchor_valid)
        support = kl_mask(forecast, model.geometry, anchors=anchors, anchor_valid=anchor_valid, validate=False)

        # Source flags: anchor step and per-channel invalid share over the 38 key steps.
        quality = out["source_quality"]
        inv_cs = torch.cat([torch.zeros_like(quality[:, :1]), torch.cumsum(1.0 - quality, dim=1)], dim=1)
        lo = (anchors - (N_LAGS - 1)).clamp_min(0)
        win = (gather_steps(inv_cs, anchors + 1) - gather_steps(inv_cs, lo)) / N_LAGS

        at = lambda x: gather_steps(x, anchors)  # noqa: E731
        per_anchor_t = {
            "key_mask": at(out["source_key_mask"]),
            "quality_anchor": at(quality) > 0.5,
            "quality_inv_share": win,
            "attn": at(out["attn_weights"]),
            "invalid_mass": at(out["source_invalid_mass"]),
            "mu_prior": mu_prior_a,
            "logvar_prior": at(out["logvar_prior"]),
            "mu_post": mu_post_a,
            "logvar_post": at(out["logvar_post"]),
            "kl_dim": at(kl),
            "kl_head": at(out["kld_per_t_per_head"]),
            "kl_lag": at(out["source_kl_lag_map"]),
        }
        keep = anchor_valid.cpu().numpy()
        for name, tensor in per_anchor_t.items():
            arrs[name].append(tensor.float().cpu().numpy()[keep].astype(ARRAYS[name][0]))

        scalars_t = {
            "gate": at(gate)[..., 0],
            "gate_logit": at(logit_heads + logit_inv)[..., 0],
            "gate_logit_heads": at(logit_heads)[..., 0],
            "gate_logit_invalid": at(logit_inv)[..., 0],
            "kl": at(out["kld_per_t"]),
            "kl_support": at(support.float()),
            "n_keys_usable": at(out["source_key_mask"].float()).sum(-1),
            "n_inv_ch_anchor": (1.0 - at(quality)).sum(-1),
            "flag_inv_share_win": win.mean(-1),
            "src_adapter_rms": at(rms(src_tap.out[0])),
            "src_adapter_prenorm_rms": at(rms(src_tap.pre[0])),
            "tgt_adapter_rms": at(rms(tgt_tap.out[0])),
            "tgt_adapter_prenorm_rms": at(rms(tgt_tap.pre[0])),
            "h_y_rms": at(rms(out["target_state"])),
            "dmu_rms": rms(mu_post_a - mu_prior_a),
        }
        for prefix, sc in (("base", sc_b), ("full", sc_f), ("base_samp", sc_bs), ("full_samp", sc_fs)):
            for key, value in sc.items():
                if prefix.endswith("samp") and key != "nll":
                    continue
                if key in ("n_st", "n_ph") and prefix != "base":
                    continue
                name = key if key in ("n_st", "n_ph") else f"{key}_{prefix}"
                scalars_t[name] = value
        for name, tensor in scalars_t.items():
            add(name, tensor.float().cpu().numpy()[keep])

        anc_np = anchors.cpu().numpy()
        guids = [S.guid_text(g) for g in batch["guid"]]
        epochs = batch["epoch"].float().cpu().numpy()
        tlo = batch["time_from_labor_onset"].float().cpu().numpy()
        sso = batch["second_stage_onset"].float().cpu().numpy()
        cs_lab = batch["cs_label"].cpu().numpy().astype(np.int8)
        bg_lab = batch["bg_label"].cpu().numpy().astype(np.int8)
        cell_row = np.full((B, A), -1, dtype=np.int64)
        for i in range(B):
            cohort, row = lookup[(guids[i], int(np.rint(epochs[i])))]
            valid = keep[i]
            t_i = anc_np[i][valid]
            n = len(t_i)
            add("cohort", np.repeat(cohort, n))
            add("membership", np.repeat(S.MEMBERSHIP[cohort], n))
            add("guid", np.repeat(guids[i], n))
            add("split_code", np.repeat(S.guid_hash(guids[i]), n).astype(np.int8))
            add("shard_row", np.repeat(row, n).astype(np.int32))
            add("epoch", np.repeat(epochs[i], n))
            add("anchor_step", t_i.astype(np.int16))
            add("sec_to_delivery", -(epochs[i] + TRIM_RAW / S.FS + S.STEP_S * t_i))
            add("time_from_labor_onset", np.repeat(tlo[i], n))
            add("second_stage_onset", np.repeat(sso[i], n))
            add("cs_label", np.repeat(cs_lab[i], n))
            add("bg_label", np.repeat(bg_lab[i], n))
            q = handles[cohort]["up_quality"][row][TRIM_RAW:-TRIM_RAW]
            for name, value in raw_validity_rows(q, t_i).items():
                add(name, value)
            pos = np.flatnonzero(valid & np.isin(anc_np[i] - floor_step, sel_pos))
            cell_row[i, pos] = cell_cursor + np.arange(len(pos))
            cell_cursor += len(pos)
        add("cell_row", cell_row[keep])

        # Per-cell arrays for the subsample, including the z = 0 and z = batch-mean decodes.
        # The dense geometry gives every row the same anchor columns, so the subsample is a set of
        # columns and the two extra decodes run on those columns only.
        sel_np = cell_row >= 0
        sel_cols = np.flatnonzero(sel_np[0])
        if sel_cols.size:
            assert (sel_np == sel_np[0][None]).all(), "subsample columns differ between rows"
            cols_t = torch.as_tensor(sel_cols, device=dev)
            zbar = mu_prior_a[anchor_valid].mean(0)
            pers = out["persistence"][:, cols_t]
            sub = mu_prior_a[:, cols_t]
            mu_z0, _ = model.decoder(torch.zeros_like(sub), persistence=pers)
            mu_zb, _ = model.decoder(zbar.expand_as(sub).contiguous(), persistence=pers)
            rows = cell_row[sel_np]
            lo_row, hi_row = int(rows.min()), int(rows.max()) + 1
            assert hi_row - lo_row == len(rows)
            for name, tensor in (("mu_full", mu_f[:, cols_t]), ("lv_full", lv_f[:, cols_t]),
                                 ("mu_base", mu_b[:, cols_t]), ("lv_base", lv_b[:, cols_t]),
                                 ("mu_z0", mu_z0), ("mu_zbar", mu_zb), ("target", target[:, cols_t]),
                                 ("step_mask", step_mask[:, cols_t] > 0.5)):
                block = tensor.reshape(-1, *tensor.shape[2:]).cpu().numpy()
                cells[name][lo_row:hi_row] = block.astype(cells[name].dtype)
        if bi % 20 == 0:
            print(f"batch {bi}: {sum(len(c) for c in cols['gate'])} anchors, {time.time() - t0:.0f} s", flush=True)

    src_tap.remove()
    tgt_tap.remove()
    for handle in handles.values():
        handle.close()
    for array in cells.values():
        array.flush()

    frame = pd.DataFrame({name: np.concatenate(parts) for name, parts in cols.items()})
    frame["G"] = frame["nll_base"] - frame["nll_full"]
    frame["G_samp"] = frame["nll_base_samp"] - frame["nll_full_samp"]
    frame.to_parquet(out_dir / "anchors.parquet", index=False)
    for name, parts in arrs.items():
        np.save(out_dir / "arrays" / f"{name}.npy", np.concatenate(parts))

    checks["n_anchors"] = int(len(frame))
    checks["n_cell_rows"] = int(cell_cursor)
    checks["n_cell_rows_allocated"] = int(n_cell_total)
    checks["n_recordings"] = int(frame["guid"].nunique())
    checks["n_segments"] = int(frame.groupby(["guid", "epoch"]).ngroups)
    checks["by_cohort_code"] = {f"{c}/{k}": int(v) for (c, k), v in
                                frame.groupby(["cohort", "split_code"])["guid"].nunique().items()}
    schema = {
        "rows": "one valid anchor of the dense evaluation geometry (stride 1, phase 0)",
        "columns": {name: str(frame[name].dtype) for name in frame.columns},
        "arrays": {name: {"dtype": d, "shape": ["N", *s], "meaning": m} for name, (d, s, m) in ARRAYS.items()},
        "cells": {name: {"dtype": d, "shape": ["N_cell", H] if name == "step_mask" else ["N_cell", H, C],
                         "meaning": m} for name, (d, m) in CELLS.items()},
        "cell_every": every,
        "anchor_floor": floor_step,
        "anchor_ceiling": ceiling,
        "ar_coef": lk["ar_coef"].cpu().tolist(),
        "cell_mask": lk["cell_mask"].cpu().numpy().astype(int).tolist(),
        "target_keep_index": model.target_gate.keep_index.cpu().tolist(),
        "source_keep_index": model.source_gate.keep_index.cpu().tolist(),
        "n_st": N_ST,
        "floor_logvar": FLOOR_LOGVAR,
        "timeline": "model step t = stored step t+15; anchor time = epoch + 60 + 4t s relative to delivery",
        "checks": checks,
    }
    (out_dir / "schema.json").write_text(json.dumps(schema, indent=1))
    print(json.dumps(checks, indent=1))
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main(vars(build_parser().parse_args()))
