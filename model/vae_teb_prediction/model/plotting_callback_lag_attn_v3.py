r"""Training-time diagnostic plots for :class:`SeqVaeLagAttnV3` (S4-T09).

**Decision and rationale.** v3 emits every forward key v1 does, so v1's 12-row diagnostic
figure renders unchanged and is reused verbatim through
:func:`~model.vae_teb_prediction.model.plotting_callback_lag_attn_v1._build_diagnostic_figure`
rather than being forked -- that builder is ~500 lines and hard-requires a fixed key set, so
copying it to add rows would be pure duplication and a drift hazard.

What v1's figure *cannot* show is the three signals v3 exists to produce, so this callback adds
a small companion figure per sample:

1. **Learned predictive band (G7).** :math:`\mu_{\mathrm{full}} \pm 2\sigma_{\mathrm{full}}`
   against the true future :math:`Y^{+}` at one anchor. Under ``sigma_obs='learned'`` the
   decoder's log-variance heads are trained, so the band is meaningful; watch it collapse
   toward the :math:`e^{-5}` floor if the variance degenerates.
2. **Per-dim raw KL and the active fraction (G4).** Which latent dimensions actually carry
   source information, against the ``kld_active_frac`` threshold. Only ``kld_per_t`` /
   ``kld_raw`` -- never the free-bit-floored ``kld_train`` -- may be read as a TE surrogate.
3. **The negative control (G6).** Per-step :math:`K_{\mathrm{true}}` against
   :math:`K_{\mathrm{shuffled}}`, the KL obtained when the source stream is deranged across
   the batch. If the two curves coincide, the reported KL is *not* source-specific and no
   amount of :math:`\beta` tuning will make it a transfer entropy.

Both figures come from a single forward pass. The callback never raises into the training
loop: like v1's, it wraps generation in a broad try/except and only warns.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
from loguru import logger

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from model.vae_teb_prediction.model.plotting_callback_lag_attn_v1 import (  # noqa: E402
    COLOR_BLACK,
    COLOR_BLUE,
    COLOR_GRAY,
    COLOR_ORANGE,
    COLOR_VERMILLION,
    LagAttnV1PlotCallback,
    _build_diagnostic_figure,
    _get_field,
    _guid_of,
    _np,
)
from utils.style import SAVE_DPI, save_figure, style_axes  # noqa: E402

_KLD_ACTIVE_EPS = 1e-2  # mirrors SeqVaeLagAttnV3._KLD_ACTIVE_EPS


def _future_target(y_st: torch.Tensor, y_ph: torch.Tensor, horizon: int) -> torch.Tensor:
    r"""Unfold :math:`Y^{+}`: at anchor :math:`t`, the window ``Y[t+1 : t+1+H]``.

    Args:
        y_st: FHR scattering features ``(B, T, 43)``.
        y_ph: FHR phase features ``(B, T, 44)``.
        horizon: Forecast horizon :math:`H_d`.

    Returns:
        ``(B, T - H_d, H_d, C_y)``, matching ``TestRunner.build_future_target``.
    """
    Y = torch.cat([y_st, y_ph], dim=-1)
    return Y[:, 1:, :].unfold(dimension=1, size=horizon, step=1).permute(0, 1, 3, 2)


def _build_v3_companion_figure(
    *,
    outs: dict,
    y_st: torch.Tensor,
    y_ph: torch.Tensor,
    kld_shuffled_per_t: torch.Tensor,
    sample_idx: int,
    epoch: int,
    guid: str,
    warmup: int,
    horizon: int,
    forecast_channels: Tuple[int, ...],
    forecast_anchor_frac: float,
    kld_active_frac: float,
    kld_shuffled_scalar: float,
) -> Any:
    """Build the three-panel v3-specific companion figure for one sample."""
    s = sample_idx
    T = int(y_st.shape[1])
    anchor = int(np.clip(round(forecast_anchor_frac * T), warmup, max(T - horizon - 1, warmup)))

    y_plus = _future_target(y_st, y_ph, horizon)  # (B, T-H, H, C)
    mu_full = _np(outs["mu_full"][s, anchor])  # (H, C)
    sigma_full = np.exp(0.5 * _np(outs["logvar_full"][s, anchor]))
    truth = _np(y_plus[s, anchor])  # (H, C)

    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(2, len(forecast_channels), height_ratios=[1.0, 1.0], hspace=0.45)

    # --- Row 1: learned predictive band, one panel per channel (G7) ----------
    steps = np.arange(1, horizon + 1)
    for i, ch in enumerate(forecast_channels):
        ax = fig.add_subplot(grid[0, i])
        if ch >= mu_full.shape[-1]:
            ax.set_visible(False)
            continue
        mu_c, sd_c, y_c = mu_full[:, ch], sigma_full[:, ch], truth[:, ch]
        ax.fill_between(steps, mu_c - 2 * sd_c, mu_c + 2 * sd_c, color=COLOR_BLUE,
                        alpha=0.22, linewidth=0, label=r"$\mu \pm 2\sigma$")
        ax.plot(steps, mu_c, color=COLOR_BLUE, linewidth=1.6, label=r"$\mu_{\mathrm{full}}$")
        ax.plot(steps, y_c, color=COLOR_VERMILLION, linewidth=1.4, linestyle="--",
                label=r"$Y^{+}$")
        cover = float(np.mean(np.abs(y_c - mu_c) <= 2 * sd_c))
        ax.set_title(f"channel {ch} @ anchor {anchor}  (2$\\sigma$ coverage {cover:.0%})")
        ax.set_xlabel("horizon step $h$")
        ax.set_ylabel("feature value" if i == 0 else "")
        style_axes(ax)
        if i == 0:
            ax.legend(loc="best", frameon=False, fontsize=8)

    # --- Row 2 left: per-dim raw KL + active threshold (G4) ------------------
    ax = fig.add_subplot(grid[1, 0])
    mu_p, lv_p = outs["mu_prior"][s], outs["logvar_prior"][s]
    mu_q, lv_q = outs["mu_post"][s], outs["logvar_post"][s]
    per_dim = _np(
        0.5 * (lv_p - lv_q + (lv_q.exp() + (mu_q - mu_p).pow(2)) / lv_p.exp() - 1.0)
    )[warmup:].mean(axis=0)
    dims = np.arange(per_dim.shape[0])
    colors = [COLOR_ORANGE if v > _KLD_ACTIVE_EPS else COLOR_GRAY for v in per_dim]
    ax.bar(dims, per_dim, color=colors, edgecolor=COLOR_BLACK, linewidth=0.4)
    ax.axhline(_KLD_ACTIVE_EPS, color=COLOR_VERMILLION, linestyle=":", linewidth=1.2,
               label=rf"active threshold $\epsilon={_KLD_ACTIVE_EPS}$")
    ax.set_yscale("symlog", linthresh=_KLD_ACTIVE_EPS)
    ax.set_title(f"raw per-dim KL   (active fraction {kld_active_frac:.2f})")
    ax.set_xlabel("latent dim $j$")
    ax.set_ylabel(r"$\overline{K_j}$ [nats]")
    ax.legend(loc="best", frameon=False, fontsize=8)
    style_axes(ax)

    # --- Row 2 rest: K_true vs K_shuffled per step (G6) -----------------------
    ax = fig.add_subplot(grid[1, 1:])
    k_true = _np(outs["kld_per_t"][s])
    k_shuf = _np(kld_shuffled_per_t[s])
    t_axis = np.arange(T)
    ax.plot(t_axis, k_true, color=COLOR_BLUE, linewidth=1.4, label=r"$K_{\mathrm{true}}$")
    ax.plot(t_axis, k_shuf, color=COLOR_VERMILLION, linewidth=1.2, alpha=0.85,
            label=r"$K_{\mathrm{shuffled}}$ (deranged UP)")
    ax.axvspan(0, warmup, color=COLOR_GRAY, alpha=0.15, linewidth=0)
    ax.axvspan(max(T - horizon, 0), T, color=COLOR_GRAY, alpha=0.15, linewidth=0)
    in_support = k_true[warmup:max(T - horizon, warmup)]
    k_true_mean = float(np.mean(in_support)) if in_support.size else 0.0
    ratio = kld_shuffled_scalar / max(k_true_mean, 1e-8)
    ax.set_title(
        "source-permutation control  "
        rf"($K_{{\mathrm{{shuffled}}}}/K_{{\mathrm{{raw}}}} \approx {ratio:.2f}$; "
        "shaded = outside the training KL support)"
    )
    ax.set_xlabel("time step $t$")
    ax.set_ylabel(r"$K_t$ [nats]")
    ax.legend(loc="best", frameon=False, fontsize=8)
    style_axes(ax)

    fig.suptitle(
        f"SeqVaeLagAttnV3 diagnostics — epoch {epoch}, sample {sample_idx}, guid {guid[:16]}",
        fontsize=12,
    )
    return fig


class LagAttnV3PlotCallback(LagAttnV1PlotCallback):
    """v1's diagnostic figure plus a v3 companion figure (G4, G6, G7).

    Args:
        output_dir: Directory under which to write ``lag_attn_v3_diagnostics``.
        plot_frequency: Plot every N validation epochs.
        num_examples: Number of samples from the first validation batch.
        file_format: Output image format (``"pdf"`` or ``"png"``).
        mlflow_logger: Optional MLflow logger; each saved file becomes a run artifact.
        forecast_channels: Channels shown in the predictive-band row.
        forecast_anchor_frac: Fractional position of the anchor whose forecast window is drawn.
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        plot_frequency: int = 5,
        num_examples: int = 2,
        *,
        file_format: str = "pdf",
        mlflow_logger: Any = None,
        forecast_channels: Tuple[int, ...] = (0, 43, 80),
        forecast_anchor_frac: float = 0.6,
    ) -> None:
        super().__init__(
            output_dir,
            plot_frequency=plot_frequency,
            num_examples=num_examples,
            file_format=file_format,
            mlflow_logger=mlflow_logger,
            forecast_channels=forecast_channels,
            forecast_anchor_frac=forecast_anchor_frac,
        )
        # Retarget the output directory; drop the empty v1 folder the parent just made.
        try:
            self.output_dir.rmdir()
        except OSError:
            pass
        self.output_dir = Path(output_dir) / "lag_attn_v3_diagnostics"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def _generate_plots(self, batch: Any, pl_module: Any, epoch: int) -> None:
        """One forward pass; write v1's 12-row figure and the v3 companion per sample."""
        batch = pl_module.transfer_batch_to_device(batch, pl_module.device, dataloader_idx=0)
        model = pl_module.orig_model

        y_st, y_ph = batch.fhr_st, batch.fhr_ph
        up_ph = _get_field(batch, "up_ph")
        if up_ph is None:
            logger.warning("LagAttnV3PlotCallback: batch has no `up_ph`; skipping.")
            return

        up_st: Optional[torch.Tensor] = None
        if bool(getattr(model, "use_up_st", False)):
            up_st = _get_field(batch, "up_st")
            if up_st is None:
                logger.warning("LagAttnV3PlotCallback: use_up_st=True but no `up_st`; skipping.")
                return
            u_stream = torch.cat([up_st, up_ph], dim=-1)
        else:
            u_stream = up_ph

        was_training = pl_module.training
        pl_module.eval()
        try:
            outs = model(y_st, y_ph, u_stream)
            beta = float(pl_module.hparams.get("kld_beta", 0.0))
            lambda_full = float(pl_module.hparams.get("lambda_full", 1.0))
            lambda_base = float(pl_module.hparams.get("lambda_base", 0.5))
            loss_dict = model.compute_loss(
                forward_outputs=outs, y_st=y_st, y_ph=y_ph, beta=beta,
                lambda_full=lambda_full, lambda_base=lambda_base,
            )
            feat_loss = float(loss_dict["feat_loss"])
            base_loss = float(loss_dict["base_loss"])
            kld_loss = float(loss_dict["kld_loss"])
            kld_active_frac = float(loss_dict.get("kld_active_frac", 0.0))

            warmup = int(getattr(model, "warmup_period", 0))
            horizon = int(getattr(model, "horizon", 30))
            step_seconds = float(getattr(model, "step_seconds", 4.0))
            delta_up_seconds = float(getattr(model, "delta_up_seconds", 0.0))

            # G6 negative control. Reuses the states from the forward above -- no re-encode.
            perm_out = None
            if y_st.shape[0] >= 2:
                gen = torch.Generator().manual_seed(epoch)
                perm_out = model.perm_kl_from_forward(outs, generator=gen)

            num_samples = min(self.num_examples, y_st.shape[0])
            for s in range(num_samples):
                guid = _guid_of(batch, s)
                fig = _build_diagnostic_figure(
                    outs=outs, y_st=y_st, y_ph=y_ph, up_st=up_st, up_ph=up_ph,
                    fhr_raw=_get_field(batch, "fhr"), up_raw=_get_field(batch, "up"),
                    sample_idx=s, epoch=epoch, guid=guid, warmup=warmup, horizon=horizon,
                    forecast_channels=self.forecast_channels,
                    forecast_anchor_frac=self.forecast_anchor_frac,
                    beta=beta, feat_loss=feat_loss, base_loss=base_loss, kld_loss=kld_loss,
                    step_seconds=step_seconds, delta_up_seconds=delta_up_seconds,
                )
                path = self.output_dir / (
                    f"lag_attn_v3_epoch{epoch:04d}_sample{s}_{guid[:16]}.{self.file_format}"
                )
                save_figure(fig, path, dpi=SAVE_DPI, close=True)
                self._log_artifact(path)

                if perm_out is None:
                    continue
                companion = _build_v3_companion_figure(
                    outs=outs, y_st=y_st, y_ph=y_ph,
                    kld_shuffled_per_t=perm_out["kld_shuffled_per_t"],
                    sample_idx=s, epoch=epoch, guid=guid, warmup=warmup, horizon=horizon,
                    forecast_channels=self.forecast_channels,
                    forecast_anchor_frac=self.forecast_anchor_frac,
                    kld_active_frac=kld_active_frac,
                    kld_shuffled_scalar=float(perm_out["kld_shuffled"]),
                )
                path = self.output_dir / (
                    f"lag_attn_v3_epoch{epoch:04d}_sample{s}_{guid[:16]}"
                    f"_control.{self.file_format}"
                )
                save_figure(companion, path, dpi=SAVE_DPI, close=True)
                self._log_artifact(path)

            logger.info(
                f"LagAttnV3PlotCallback: saved {num_samples} figure pair(s) for epoch "
                f"{epoch} to {self.output_dir}"
            )
        finally:
            if was_training:
                pl_module.train()
