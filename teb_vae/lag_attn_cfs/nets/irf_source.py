"""The UA -> FHR impulse-response (IRF) head of ``source_path='irf'``. Method: teb_vae/coupling_irf/DESIGN.md.

``teb_vae.coupling_irf.model.CouplingIRF`` ported to C target channels. Times are in steps of 4 s on the
stored clock. The head adds a mean shift to a forecast it is given:

    delta(t, tau, c) = a[t, c] * sum_j h[t, j, c] * (eta[tau, c] * Xf[t, tau, j] - rho[tau, c] * Z[t, j])

u          UA level above its causal resting tone, floored at 0.
Z(t)       u convolved with J hat functions over delays 0..120 s (the IRF is zero at 0 and at 120 s):
           what UA can explain at step t.
Xf(t, tau) the same at step t + tau, with UA up to the anchor t only. tau = 1..H.
h_t        ridge IRF per channel, fitted on the past of the same sequence after the same causal
           high-pass on the target and on Z.
a_t        how much of h_t to use, in [0, 1], per channel: the slope that calibrates the IRF's
           out-of-sample predictions of the past target, minus k standard errors. With a UA that does
           not belong to the FHR it is 0.

The head has no network. Its trained numbers are two penalties and rho / eta per horizon step and channel.
"""
import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from teb_vae.coupling_irf.model import KNOTS, hat_basis, trailing_quantile, trailing_sum

_SOLVE_ON_CPU = False  # set once, by the first solve that finds the GPU has no batched solver


def batched_solve(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Solve A X = B on the tensors' device, or on the CPU once that device has refused the kernel."""
    global _SOLVE_ON_CPU
    if not _SOLVE_ON_CPU:
        try:
            return torch.linalg.solve(A, B)
        except RuntimeError as err:
            if "no kernel image" not in str(err):
                raise
            _SOLVE_ON_CPU = True
    return torch.linalg.solve(A.cpu(), B.cpu()).to(A.device)


def ua_step_level(up: torch.Tensor, up_valid: torch.Tensor, decimation: int, steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mean UA and validity per step, from the trimmed raw trace ``up`` (B, L) and its mask ``up_valid``.

    Trimmed step t ends at raw sample ``decimation * t`` and covers the ``decimation`` samples that end
    there (``hdf5_dataset._step_masks``), so step 0 has one sample inside the trimmed window. A step is
    valid if at least half of its samples are.

    Returns:
        ``(level (B, steps), valid (B, steps) bool)``.
    """
    front = decimation - 1
    v = F.pad(up_valid.to(torch.bool), (front, 0))[:, : decimation * steps].reshape(-1, steps, decimation)
    x = F.pad(up, (front, 0))[:, : decimation * steps].reshape(-1, steps, decimation)
    n = v.sum(-1)
    return torch.where(v, x, torch.zeros_like(x)).sum(-1) / n.clamp_min(1), n >= decimation / 2


class IrfSourceHead(nn.Module):
    TONE_WINDOW = 75      # steps of the trailing 10th percentile that is the resting tone
    HIGHPASS_WINDOW = 75  # steps of the trailing mean removed from the target and from Z
    PREQ_GAP = 15         # the IRF that predicts step s was fitted on steps <= s - PREQ_GAP
    MIN_UA_COVER = 0.8    # valid share of the last 120 s of UA for Z to count
    MIN_PREQ_ROWS = 10    # prequential rows in the window before the gate can open

    def __init__(self, horizon: int, channels: int, gate_k: float = 4.4, context_steps: int = 0, knots=KNOTS):
        super().__init__()
        self.H, self.D, self.J = int(horizon), int(knots[-1]), len(knots) - 2
        self.gate_k, self.context_steps = float(gate_k), int(context_steps)
        _, w_full, w_fore, pen_smooth = hat_basis(self.H, knots)
        # Functions of the knots and the horizon alone, so not in the state dict.
        self.register_buffer("w_full", w_full, persistent=False)                                    # (J, 1, D+1)
        self.register_buffer("w_fore", w_fore.reshape(self.H, self.J, self.D + 1), persistent=False)
        self.register_buffer("pen_smooth", pen_smooth.float(), persistent=False)
        self.log_lambda = nn.Parameter(torch.full((2,), math.log(0.3)))      # per-row ridge and smoothness penalties
        self.rho_logit = nn.Parameter(torch.zeros(self.H, channels))
        self.eta_logit = nn.Parameter(torch.full((self.H, channels), 2.0))

    def context_irf(self, Z, z_ok, y, y_ok, step_ok) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""At every step: the ridge IRF of each channel ``h`` (B, T, J, C), its gate ``a`` (B, T, C)
        and the gate's statistic ``z`` = slope / standard error (B, T, C); the gate is open where z > k.

        The normal equations and the prequential sums run in float64, as in
        ``teb_vae.coupling_irf.model``, and the three outputs are cast back to the input dtype. A
        prequential row $s$ counts only when the IRF fitted at $s - g$ had rows of its own
        ($g$ = ``PREQ_GAP``), so the startup steps with no prediction yet add neither residual nor
        support. The gate stays shut ($a = z = 0$) until ``MIN_PREQ_ROWS`` such rows are in the window.
        """
        W, g, hp = self.context_steps, self.PREQ_GAP, self.HIGHPASS_WINDOW
        dtype = Z.dtype
        Z, y = Z.double(), y.double()
        # ponytail: one normal matrix for all channels, with rows w = UA covered and FHR step valid.
        # A channel's own invalid rows (its warm-up) only zero its residual, so its IRF is shrunk a
        # little and its two trailing means cover different rows. Use per-channel matrices if slow
        # channels with long warm-ups matter.
        w = (z_ok & step_ok).to(Z.dtype)
        v = w[..., None] * y_ok.to(Z.dtype)                                  # (B, T, C) rows of each channel
        y = torch.where(y_ok, y, torch.zeros_like(y))
        # the same causal high-pass (minus trailing mean) on both sides keeps the regression unbiased
        cnt, cnt_c = trailing_sum(w, hp), trailing_sum(v, hp)
        mZ = trailing_sum(w[..., None] * Z, hp) / cnt.clamp_min(1)[..., None]
        my = trailing_sum(v * y, hp) / cnt_c.clamp_min(1)
        w = w * (cnt >= 10)                                                  # rows with a usable trailing mean
        v = v * w[..., None] * (cnt_c >= 10)
        Zc, r = (Z - mZ) * w[..., None], (y - my) * v
        Szz = trailing_sum(Zc[..., :, None] * Zc[..., None, :], W)           # (B, T, J, J)
        Szr = trailing_sum(Zc[..., :, None] * r[..., None, :], W)            # (B, T, J, C)
        lam = torch.exp(self.log_lambda).double()
        pen = lam[0] * torch.eye(self.J, dtype=Z.dtype, device=Z.device) + lam[1] * self.pen_smooth.double()
        h = batched_solve(Szz + trailing_sum(w, W).clamp_min(1)[..., None, None] * pen, Szr)
        # prequential check: predict step s with the IRF fitted on steps <= s - gap, then calibrate
        h_past = torch.cat([torch.zeros_like(h[:, :g]), h[:, :-g]], 1)
        fitted = trailing_sum(v, W)                                          # (B, T, C) rows each IRF saw
        vp = v * (torch.cat([torch.zeros_like(fitted[:, :g]), fitted[:, :-g]], 1) > 0)
        pred, rp = torch.einsum("btjc,btj->btc", h_past, Zc) * vp, r * vp
        Spp, Spr, n = trailing_sum(pred * pred, W), trailing_sum(pred * rp, W), trailing_sum(vp, W)
        slope = Spr / (Spp + 1e-9)
        resid = (trailing_sum(rp * rp, W) - 2 * slope * Spr + slope * slope * Spp).clamp_min(0) / n.clamp_min(1)
        se = torch.sqrt((resid + 1e-12) / (Spp + 1e-9))
        enough = n >= self.MIN_PREQ_ROWS
        a = torch.where(enough, (slope - self.gate_k * se).clamp(0, 1), torch.zeros_like(slope))
        z = torch.where(enough, slope / se, torch.zeros_like(slope))
        return h.to(dtype), a.to(dtype), z.to(dtype)

    def forward(self, y, y_ok, step_ok, ua, ua_ok, anchors) -> Dict[str, torch.Tensor]:
        """The mean shifts at the anchors.

        Args:
            y: Target features (B, T, C), the tensor the likelihood scores.
            y_ok: Their validity (B, T, C), bool.
            step_ok: FHR step validity (B, T), bool, the same for all channels.
            ua: UA level per step (B, T).
            ua_ok: Its validity (B, T), bool.
            anchors: Anchor steps (B, A), long, below T - H.

        Returns:
            ``delta`` (B, A, H, C), the shift that uses UA up to the anchor. ``delta_expl`` (B, A, H, C),
            the shift that uses UA up to the target step: a readout, never a forecast. ``h`` (B, A, J, C)
            and ``a`` (B, A, C). ``z`` (B, A, C), the gate's slope over its standard error: the gate
            is open where ``z > gate_k``, which is what a calibration of ``gate_k`` reads. ``ua_ok``
            (B, A): UA covers the last 120 s at the anchor; ``a`` and ``z`` are 0 where it does not,
            and ``delta_expl`` is 0 where UA does not cover the target step. ``evidence`` (B, T, 2),
            at every step and not only at the anchors: ``clamp(z / gate_k, 0, 3)`` of the first
            channel, and the share of the channels whose gate is open; both 0 without UA cover.
        """
        B, T = ua.shape
        H, D = self.H, self.D
        tone = trailing_quantile(ua, ua_ok, self.TONE_WINDOW, 0.1)
        u = torch.where(ua_ok & torch.isfinite(tone), (ua - tone).clamp_min(0), torch.zeros_like(ua))
        pad = F.pad(u[:, None], (D, 0))
        Z = F.conv1d(pad, self.w_full).transpose(1, 2)                       # (B, T, J)
        cover = F.avg_pool1d(F.pad(ua_ok.to(u.dtype)[:, None], (D, 0)), D + 1, 1)[:, 0]
        z_ok = (cover >= self.MIN_UA_COVER) & (torch.arange(T, device=ua.device)[None] >= D)
        h, a, z = self.context_irf(Z, z_ok, y, y_ok, step_ok)
        evidence = torch.stack([(z[..., 0] / self.gate_k).clamp(0, 3), (a > 0).to(z.dtype).mean(-1)], -1)
        evidence = evidence * z_ok[..., None]

        rows, t = torch.arange(B, device=ua.device)[:, None], anchors.to(torch.long)
        tgt = t[..., None] + 1 + torch.arange(H, device=ua.device)           # (B, A, H) target steps
        h, ok = h[rows, t], z_ok[rows, t]
        a = a[rows, t] * ok[..., None]
        beta = a[:, :, None] * h                                             # (B, A, J, C)
        z_now = torch.einsum("bajc,baj->bac", beta, Z[rows, t])[:, :, None]  # what UA explains at the anchor
        Xf = torch.einsum("bad,hjd->bahj", pad[:, 0].unfold(1, D + 1, 1)[rows, t], self.w_fore)
        rho, eta = torch.sigmoid(self.rho_logit), torch.sigmoid(self.eta_logit)
        delta = eta * torch.einsum("bajc,bahj->bahc", beta, Xf) - rho * z_now
        delta_expl = torch.einsum("bajc,bahj->bahc", beta, Z[rows[..., None], tgt]) - rho * z_now
        return dict(delta=delta, delta_expl=delta_expl * z_ok[rows[..., None], tgt][..., None], h=h, a=a,
                    z=z[rows, t] * ok[..., None], ua_ok=ok, evidence=evidence)
