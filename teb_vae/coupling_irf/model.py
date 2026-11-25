"""UA -> FHR coupling as an impulse response (IRF) estimated in context, on the 4-s grid. Design: DESIGN.md.

Times are in steps of 4 s on the stored clock (UA is stored 20 s earlier than it was recorded).
FHR is in units of 20 bpm around 140 bpm; UA is toco units above a causal resting tone, divided by 25.

    full forecast(t, tau) = base forecast(t, tau) + beta_t . (X(t, tau) - rho_tau * Z(t))

base      FHR-only forecaster (causal TCN on the FHR level); never sees UA.
Z(t)      UA convolved with J hat functions over delays 0..120 s (the IRF is zero at 0 and at 120 s):
          what UA can explain at step t.
X(t, tau) the same at step t + tau. "fore" mode uses UA up to the anchor t only; "expl" mode uses UA up
          to the target t + tau.
h_t       ridge IRF fitted on the recording's own past (the last `context_steps` steps up to t), after
          the same causal high-pass on FHR and on Z.
a_t       how much of h_t to use, in [0, 1]: the slope that calibrates the IRF's out-of-sample predictions
          of the past FHR, minus k standard errors. With a UA that does not belong to the FHR it is 0.
beta_t    a_t * h_t.

The UA path has no network: its only trained numbers are two penalties, k, and rho / eta per horizon.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

STEP_S = 4.0
FHR_CENTER, FHR_SCALE, UA_SCALE = 140.0, 20.0, 25.0
T_NU = 8.0  # Student-t degrees of freedom, as in the lag-attention models
KNOTS = tuple(range(0, 31))  # delays in steps: every 4 s from 0 to 120 s


def trailing_quantile(x, ok, window, q):
    """Causal running quantile of the valid samples of x (B, T) over the last `window` steps. NaN if none."""
    x = torch.where(ok, x, torch.full_like(x, float("nan")))
    x = F.pad(x, (window - 1, 0), value=float("nan")).unfold(1, window, 1)
    return torch.nanquantile(x, q, dim=-1)


def trailing_sum(x, window):
    """Sum of x (B, T, ...) over the last `window` steps (all past steps if window <= 0)."""
    c = x.cumsum(1)
    if window <= 0 or window >= x.shape[1]:
        return c
    return c - torch.cat([torch.zeros_like(c[:, :window]), c[:, :-window]], 1)


def student_t_nll(y, mu, log_sigma, nu=T_NU):
    z = (y - mu) * torch.exp(-log_sigma)
    const = math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(nu * math.pi)
    return log_sigma - const + (nu + 1) / 2 * torch.log1p(z * z / nu)


def irf_delay(h, knots=KNOTS):
    """Delay (steps) and depth of the most negative knot of IRF coefficients h (..., J), refined by the
    parabola through its two neighbours. h holds the interior knots; the IRF is zero at the first and
    last knot. NaN where the IRF has no negative point."""
    c = torch.as_tensor(knots, dtype=h.dtype, device=h.device)
    h = F.pad(h, (1, 1))
    J = h.shape[-1]
    depth, i = h.min(-1)
    lo, hi = (i - 1).clamp(0, J - 1), (i + 1).clamp(0, J - 1)
    y0, y2 = h.gather(-1, lo[..., None])[..., 0], h.gather(-1, hi[..., None])[..., 0]
    x0, x1, x2 = c[lo], c[i], c[hi]
    num = (x1 - x0) ** 2 * (depth - y2) - (x1 - x2) ** 2 * (depth - y0)
    den = (x1 - x0) * (depth - y2) - (x1 - x2) * (depth - y0)
    ok = (depth < 0) & (i > 0) & (i < J - 1) & (den.abs() > 1e-12)
    delay = x1 - 0.5 * num / torch.where(ok, den, torch.ones_like(den))
    delay = torch.minimum(torch.maximum(delay, x0), x2)
    return torch.where(ok, delay, torch.full_like(delay, float("nan"))), depth


def hat_basis(horizon, knots=KNOTS):
    """The IRF's hat functions on the step grid and what is built from them, as
    (basis, w_full, w_fore, pen_smooth): basis (J, D+1), one hat per interior knot; w_full (J, 1, D+1),
    the conv weight that gives Z; w_fore (horizon * J, 1, D+1), the same with delays < tau removed for
    tau = 1..horizon; pen_smooth (J, J) float64, the second-derivative penalty on uneven knots."""
    D, K = int(knots[-1]), len(knots)
    J = K - 2
    basis = torch.as_tensor(np.stack([np.interp(np.arange(D + 1), knots, np.eye(K)[j]) for j in range(1, K - 1)]), dtype=torch.float32)
    shifted = torch.zeros(horizon, J, D + 1)
    for tau in range(1, min(horizon, D) + 1):
        shifted[tau - 1, :, : D + 1 - tau] = basis[:, tau:]
    c = torch.as_tensor(knots, dtype=torch.float64)
    d2 = torch.zeros(K - 2, K, dtype=torch.float64)
    for j in range(1, K - 1):
        d2[j - 1, j - 1], d2[j - 1, j + 1] = 1 / (c[j] - c[j - 1]), 1 / (c[j + 1] - c[j])
        d2[j - 1, j] = -d2[j - 1, j - 1] - d2[j - 1, j + 1]
    d2 = d2[:, 1:-1]
    return basis, basis.flip(-1)[:, None], shifted.flip(-1).reshape(horizon * J, 1, D + 1), d2.T @ d2


class CausalBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.pad = 2 * dilation
        self.conv = nn.Conv1d(width, width, 3, dilation=dilation)
        self.mix = nn.Conv1d(width, width, 1)

    def forward(self, x):
        return x + self.mix(F.gelu(self.conv(F.pad(x, (self.pad, 0)))))


class CouplingIRF(nn.Module):
    def __init__(self, horizon=30, knots=KNOTS, first_anchor=60, width=64, dilations=(1, 2, 4, 8, 16, 32),
                 tone_window=75, highpass_window=75, context_steps=900, preq_gap=15, min_ua_cover=0.8):
        super().__init__()
        self.H, self.D, self.first_anchor, self.knots = horizon, int(knots[-1]), first_anchor, tuple(knots)
        self.tone_window, self.highpass_window, self.context_steps = tone_window, highpass_window, context_steps
        self.preq_gap, self.min_ua_cover = preq_gap, min_ua_cover
        self.J = J = len(knots) - 2                                          # the IRF is zero at the two end knots
        basis, w_full, w_fore, pen_smooth = hat_basis(horizon, knots)
        self.register_buffer("basis", basis)                                 # (J, D+1) hat functions
        # conv weights: Z uses the whole kernel; "fore" mode at horizon tau keeps delays >= tau only
        self.register_buffer("w_full", w_full)
        self.register_buffer("w_fore", w_fore)
        self.register_buffer("pen_smooth", pen_smooth)
        self.register_buffer("pen_ridge", torch.eye(J, dtype=torch.float64))
        self.log_lambda = nn.Parameter(torch.full((2,), math.log(0.3)))      # per-row ridge and smoothness penalties
        self.k_raw = nn.Parameter(torch.tensor(0.5413))                      # softplus -> 1 standard error
        self.rho_logit = nn.Parameter(torch.zeros(horizon))
        self.eta_logit = nn.Parameter(torch.full((horizon,), 2.0))

        # FHR-only base forecaster
        self.stem = nn.Conv1d(3, width, 1)
        self.tcn = nn.Sequential(*[CausalBlock(width, d) for d in dilations])
        self.head_mu = nn.Linear(width, horizon)
        self.head_ls = nn.Linear(width, horizon)
        nn.init.zeros_(self.head_mu.weight); nn.init.zeros_(self.head_mu.bias)

    # ---------------------------------------------------------------- FHR side (no UA)
    def fhr_side(self, fhr, fhr_sd, fhr_ok):
        """fhr, fhr_sd in bpm (B, T); fhr_ok = valid fraction of each step."""
        B, T = fhr.shape
        v = fhr_ok >= 0.5
        yn = (fhr - FHR_CENTER) / FHR_SCALE
        feats = torch.stack([yn * v, torch.log1p(fhr_sd.float()) * v, v.float()], 1)
        c = self.tcn(self.stem(feats)).transpose(1, 2)                       # (B, T, W)
        anchors = torch.arange(self.first_anchor, T - self.H, device=fhr.device)
        tgt = anchors[:, None] + torch.arange(1, self.H + 1, device=fhr.device)[None]     # (A, H)
        idx = torch.where(v, torch.arange(T, device=fhr.device)[None], torch.zeros_like(v, dtype=torch.long)).cummax(1).values
        y_last = yn.gather(1, idx)                                           # last valid level at or before t
        ca = c[:, anchors]
        mu0 = y_last[:, anchors, None] + self.head_mu(ca)
        ls0 = self.head_ls(ca).clamp(-6.0, 3.0)
        mask = v[:, anchors, None] & v[:, tgt]
        return dict(anchors=anchors, tgt=tgt, y=yn[:, tgt], mask=mask, mu0=mu0, ls0=ls0, yn=yn, v=v)

    # ---------------------------------------------------------------- UA side
    def ua_features(self, ua, ua_ok):
        ok = ua_ok >= 0.5
        tone = trailing_quantile(ua, ok, self.tone_window, 0.1)
        u = torch.where(ok & torch.isfinite(tone), (ua - tone).clamp_min(0) / UA_SCALE, torch.zeros_like(ua))
        pad = F.pad(u[:, None], (self.D, 0))
        Z = F.conv1d(pad, self.w_full).transpose(1, 2)                       # (B, T, J)
        cover = F.avg_pool1d(F.pad(ok.float()[:, None], (self.D, 0)), self.D + 1, 1)[:, 0]
        T = ua.shape[1]
        z_ok = (cover >= self.min_ua_cover) & (torch.arange(T, device=ua.device)[None] >= self.D)
        Xf = F.conv1d(pad, self.w_fore).transpose(1, 2).reshape(ua.shape[0], T, self.H, self.J)
        return dict(u=u, Z=Z, z_ok=z_ok, Xf=Xf)

    def context_irf(self, Z, z_ok, yn, v):
        """At every step: the ridge IRF of the last `context_steps` steps and how much of it to trust."""
        W, g = self.context_steps, self.preq_gap
        w = (z_ok & v).double()
        Z, yn = Z.double(), yn.double()
        # the same causal high-pass (minus trailing mean) on both sides keeps the regression unbiased
        cnt = trailing_sum(w, self.highpass_window)
        my = trailing_sum(w * yn, self.highpass_window) / cnt.clamp_min(1)
        mZ = trailing_sum(w[..., None] * Z, self.highpass_window) / cnt.clamp_min(1)[..., None]
        w = w * (cnt >= 10)                                                  # rows with a usable trailing mean
        r, Zc = (yn - my) * w, (Z - mZ) * w[..., None]
        Szz = trailing_sum(Zc[..., :, None] * Zc[..., None, :], W)           # (B, T, J, J)
        Szr, Srr, n = trailing_sum(Zc * r[..., None], W), trailing_sum(r * r, W), trailing_sum(w, W)
        lam = torch.exp(self.log_lambda).double()
        pen = lam[0] * self.pen_ridge + lam[1] * self.pen_smooth
        A = Szz + n.clamp_min(1)[..., None, None] * pen
        # ponytail: solved on the CPU (this laptop's GPU has no batched solver in the installed torch);
        # 20 x 20 systems, a fraction of a second per batch. Write an unrolled Cholesky if it limits throughput.
        h = torch.linalg.solve(A.cpu(), Szr[..., None].cpu())[..., 0].to(Z.device)       # (B, T, J)
        # prequential check: predict step s with the IRF fitted on steps <= s - gap, then calibrate
        pred = (torch.cat([torch.zeros_like(h[:, :g]), h[:, :-g]], 1) * Zc).sum(-1)
        Spp, Spr = trailing_sum(pred * pred, W), trailing_sum(pred * r, W)
        slope = Spr / (Spp + 1e-9)
        resid = (Srr - 2 * slope * Spr + slope * slope * Spp).clamp_min(0) / n.clamp_min(1)
        se = torch.sqrt((resid + 1e-12) / (Spp + 1e-9))
        a = (slope - F.softplus(self.k_raw).double() * se).clamp(0, 1)
        skill = Spr * Spr / ((Spp + 1e-9) * Srr.clamp_min(1e-9)) * torch.sign(Spr)       # signed prequential R^2
        return {k: x.float() for k, x in dict(h=h, a=a, slope=slope, se=se, skill=skill, n=n).items()}

    def ua_side(self, st, ua, ua_ok):
        """Coupling head for one UA signal against the FHR state `st`. Returns mean shifts for both modes."""
        f = self.ua_features(ua, ua_ok)
        ctx = self.context_irf(f["Z"], f["z_ok"], st["yn"], st["v"])
        a_idx, tgt = st["anchors"], st["tgt"]
        ctx = {k: x[:, a_idx] for k, x in ctx.items()}
        beta = ctx["a"][..., None] * ctx["h"]                                # (B, A, J)
        z_now = (beta * f["Z"][:, a_idx]).sum(-1)                            # what UA explains at the anchor
        rho, eta = torch.sigmoid(self.rho_logit), torch.sigmoid(self.eta_logit)
        d_fore = eta * (beta[:, :, None] * f["Xf"][:, a_idx]).sum(-1) - rho * z_now[..., None]
        d_expl = (beta[:, :, None] * f["Z"][:, tgt]).sum(-1) - rho * z_now[..., None]
        return dict(d_fore=d_fore, d_expl=d_expl, beta=beta, ua_ok=f["z_ok"][:, a_idx], ua_ok_tgt=f["z_ok"][:, tgt],
                    u=f["u"], Z=f["Z"], **ctx)

    # ---------------------------------------------------------------- likelihood
    def nll(self, st, delta=None):
        """Student-t NLL per cell (B, A, H). With `delta` the base forecast is frozen (gain = pure mean shift)."""
        if delta is None:
            return student_t_nll(st["y"], st["mu0"], st["ls0"])
        return student_t_nll(st["y"], st["mu0"].detach() + delta, st["ls0"].detach())

    def forward(self, batch, ua=None, ua_ok=None):
        st = self.fhr_side(batch["fhr"], batch["fhr_sd"], batch["fhr_ok"])
        out = self.ua_side(st, batch["ua"] if ua is None else ua, batch["ua_ok"] if ua_ok is None else ua_ok)
        return st, out


def cell_masks(st, out):
    """Scored cells per mode: FHR valid at anchor and target, UA covered at the anchor (and at the target
    for the "expl" mode, which reads UA up to the target)."""
    fore = st["mask"] & out["ua_ok"][..., None]
    return {"fore": fore, "expl": fore & out["ua_ok_tgt"]}


def gains(model, st, out):
    """Coupling per anchor in nats per valid cell: base NLL minus full NLL. NaN where no cell is valid."""
    nll0, res = model.nll(st), {}
    for mode, m in cell_masks(st, out).items():
        g = ((nll0 - model.nll(st, out["d_" + mode])) * m).sum(-1) / m.sum(-1).clamp_min(1)
        res[mode] = torch.where(m.any(-1), g, torch.full_like(g, float("nan")))
    return res


def contraction_runs(u, threshold=0.2):
    """Label the contractions of u (B, T): 0 outside, 1, 2, ... inside each run above `threshold` of the row's maximum."""
    on = u > threshold * u.amax(-1, keepdim=True).clamp_min(0.2)
    start = on & ~F.pad(on, (1, 0))[:, :-1]
    return start.long().cumsum(1) * on


def plant(fhr, u, delay, sigma=2.0, depth=20.0, share=1.0, generator=None):
    """Subtract a synthetic response from fhr (B, T) bpm: u (B, T) convolved with a unit-area Gaussian
    centred `delay` steps later, scaled so the row's largest contraction gives `depth` bpm. With share < 1
    only that random share of the contractions gets a response. delay is (B,); sigma, depth floats or (B,)."""
    K = 64
    sigma = torch.as_tensor(sigma, device=fhr.device, dtype=fhr.dtype).reshape(-1, 1)
    depth = torch.as_tensor(depth, device=fhr.device, dtype=fhr.dtype).reshape(-1, 1)
    peak = u.amax(-1, keepdim=True).clamp_min(0.2)
    if share < 1:
        runs = contraction_runs(u)
        keep = torch.rand(u.shape[0], int(runs.max()) + 1, generator=generator).to(u.device) < share
        u = u * keep.gather(1, runs) * (runs > 0)
    k = torch.exp(-0.5 * ((torch.arange(K, device=fhr.device)[None] - delay[:, None]) / sigma) ** 2)
    k = k / k.sum(-1, keepdim=True)
    resp = F.conv1d(F.pad(u[None], (K - 1, 0)), k.flip(-1)[:, None], groups=u.shape[0])[0]
    return fhr - depth * resp / peak
