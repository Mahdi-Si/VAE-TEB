r"""Raw diagnostic plotting callback for :class:`SeqVaeRawV4` training (S5-T05).

``RawV4PlotCallback`` subclasses :class:`LagAttnV3PlotCallback` -- reusing its Lightning hook
(rank-0 gate, ``plot_frequency`` schedule, the crash-proof ``try/except`` around plotting) -- but
fully re-implements :meth:`_generate_plots`, because v3's builder reads the feature-domain batch
fields (``fhr_st`` / ``fhr_ph``) and plots $87$-channel feature heatmaps that do not exist on the raw
pathway. The raw figure instead shows:

* **predicted-vs-true future raw FHR overlays in denormalized bpm** (the headline change), with the
  learned predictive band $\mu_{\mathrm{full}} \pm 2\sigma$, at a few anchors;
* per-dim KL (G4) and $K_{\mathrm{true}}$ vs $K_{\mathrm{shuffled}}$ per step (G6);
* the lag-attention matrix and the TE-lag map;
* a per-horizon raw forecast MAE row; and
* a ``pred_gap``-vs-$K$ pathology summary ($K \gg 0$ while ``pred_gap`` $\le 0$ flags a latent that
  carries information which does not improve the forecast).

Denormalization uses :func:`train.graph_models_utils.denormalize_signal_data` with the loader's
``fhr`` z-score stats (resolved best-effort from the validation dataloader); if the stats are
unavailable the overlay degrades gracefully to normalized units.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from loguru import logger  # noqa: E402

from model.vae_teb_prediction.model.model_raw.geometry import RawGeometry  # noqa: E402
from model.vae_teb_prediction.model.model_raw.raw_masks import (  # noqa: E402
    forecast_mask,
    low_rate_mask,
)
from model.vae_teb_prediction.model.model_raw.raw_targets import build_future_target  # noqa: E402
from model.vae_teb_prediction.model.plotting_callback_lag_attn_v1 import (  # noqa: E402
    COLOR_BLACK,
    COLOR_BLUE,
    COLOR_GRAY,
    COLOR_ORANGE,
    COLOR_VERMILLION,
    _guid_of,
    _np,
)
from model.vae_teb_prediction.model.plotting_callback_lag_attn_v3 import (  # noqa: E402
    LagAttnV3PlotCallback,
)
from train.graph_models_utils import denormalize_signal_data  # noqa: E402
from utils.style import SAVE_DPI, save_figure  # noqa: E402


def _denorm_fhr(x: torch.Tensor, stats: Optional[Dict[str, Any]]) -> torch.Tensor:
    """Denormalize a normalized-FHR tensor to bpm, or return it unchanged when no stats.

    ``stats`` is the full field-keyed dict (``{"fhr": {"mean":.., "std":..}}``) that
    :func:`train.graph_models_utils.denormalize_signal_data` indexes by field name.
    """
    if not stats or "fhr" not in stats:
        return x
    return denormalize_signal_data(x, "fhr", stats)


class RawV4PlotCallback(LagAttnV3PlotCallback):
    r"""Raw-domain diagnostic figures for :class:`SeqVaeRawV4`.

    Args:
        output_dir: Directory under which to write ``raw_v4_diagnostics``.
        plot_frequency: Plot every $N$ validation epochs.
        num_examples: Number of samples from the first validation batch.
        file_format: Output image format (``"pdf"`` or ``"png"``).
        mlflow_logger: Optional MLflow logger; each saved file becomes a run artifact.
        forecast_anchor_frac: Fractional position of the first anchor whose forecast window is drawn.
        denorm_stats: Optional ``{"mean":.., "std":..}`` FHR z-score stats. When ``None`` they are
            resolved best-effort from the validation dataloader on the first validation epoch.
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        plot_frequency: int = 5,
        num_examples: int = 2,
        *,
        file_format: str = "pdf",
        mlflow_logger: Any = None,
        forecast_anchor_frac: float = 0.6,
        denorm_stats: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            output_dir,
            plot_frequency=plot_frequency,
            num_examples=num_examples,
            file_format=file_format,
            mlflow_logger=mlflow_logger,
            forecast_anchor_frac=forecast_anchor_frac,
        )
        # Retarget the output directory; drop the empty v3 folder the parent just made.
        try:
            self.output_dir.rmdir()
        except OSError:
            pass
        self.output_dir = Path(output_dir) / "raw_v4_diagnostics"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._denorm_stats = denorm_stats
        self._stats_resolved = denorm_stats is not None

    # ------------------------------------------------------------------
    # Lightning hook -- resolve denorm stats once, then defer to the parent's gate.
    # ------------------------------------------------------------------
    def on_validation_epoch_end(self, trainer, pl_module) -> None:  # type: ignore[override]
        """Lazily resolve FHR denorm stats from the val dataloader, then plot as scheduled."""
        if not self._stats_resolved:
            self._denorm_stats = self._try_resolve_stats(trainer)
            self._stats_resolved = True
        super().on_validation_epoch_end(trainer, pl_module)

    @staticmethod
    def _try_resolve_stats(trainer) -> Optional[Dict[str, Any]]:
        """Best-effort FHR z-score stats from the validation dataloader (``None`` on any failure)."""
        try:
            from model.vae_teb_prediction.testing.collectors import (
                resolve_fhr_up_denorm_stats,
            )

            loaders = getattr(trainer, "val_dataloaders", None)
            if loaders is None:
                return None
            loader = loaders[0] if isinstance(loaders, (list, tuple)) else loaders
            stats = resolve_fhr_up_denorm_stats(loader)
            return stats or None
        except Exception as exc:  # noqa: BLE001 -- plotting must never crash training
            logger.debug(f"RawV4PlotCallback: could not resolve denorm stats: {exc}")
            return None

    # ------------------------------------------------------------------
    # Plot generation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _generate_plots(self, batch: Any, pl_module: Any, epoch: int) -> None:
        """One raw forward pass; write a raw diagnostic figure per sample."""
        batch = pl_module.transfer_batch_to_device(batch, pl_module.device, dataloader_idx=0)
        model = pl_module.orig_model
        geo: RawGeometry = model.geometry

        fhr_raw, up_raw, mask = model._default_batch_to_inputs(batch)

        was_training = pl_module.training
        pl_module.eval()
        try:
            outs = model(fhr_raw, up_raw, mask)

            x_plus = build_future_target(fhr_raw, geo)                 # (B, T_valid, H, R)
            m_low = low_rate_mask(mask, geo)
            f_mask = forecast_mask(mask, m_low, geo)                   # (B, T_valid, H, R)

            # G6 negative control (reuses the forward states -- no re-encode).
            perm_out = None
            if fhr_raw.shape[0] >= 2:
                gen = torch.Generator().manual_seed(epoch)
                perm_out = model.perm_kl_from_forward(outs, generator=gen)

            warmup = int(getattr(model, "warmup_period", geo.warmup))
            num_samples = min(self.num_examples, fhr_raw.shape[0])
            for s in range(num_samples):
                guid = _guid_of(batch, s)
                fig = self._build_raw_figure(
                    outs=outs, x_plus=x_plus, f_mask=f_mask, geo=geo,
                    perm_out=perm_out, sample_idx=s, epoch=epoch, guid=guid, warmup=warmup,
                )
                path = self.output_dir / (
                    f"raw_v4_epoch{epoch:04d}_sample{s}_{guid[:16]}.{self.file_format}"
                )
                save_figure(fig, path, dpi=SAVE_DPI, close=True)
                self._log_artifact(path)

            logger.info(
                f"RawV4PlotCallback: saved {num_samples} figure(s) for epoch {epoch} "
                f"to {self.output_dir}"
            )
        finally:
            if was_training:
                pl_module.train()

    def _build_raw_figure(
        self, *, outs, x_plus, f_mask, geo, perm_out, sample_idx, epoch, guid, warmup,
    ):
        r"""Assemble the $3\times3$ raw diagnostic figure for one sample.

        Rows: (0) predictive-band bpm overlays at three anchors; (1) per-dim KL, $K_{\mathrm{true}}$
        vs $K_{\mathrm{shuffled}}$ per step, per-horizon raw MAE; (2) lag-attention matrix, TE-lag
        map, and a ``pred_gap``-vs-$K$ pathology summary.
        """
        s = sample_idx
        t_valid = geo.t_valid
        horizon, r = geo.horizon, geo.r
        stats = self._denorm_stats
        unit = "bpm" if (stats and "fhr" in stats) else "norm"

        mu_full = outs["mu_full"][:, :t_valid]                        # (B, T_valid, H, R)
        logvar_full = outs["logvar_full"][:, :t_valid]
        sigma_full = torch.exp(0.5 * logvar_full)

        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        fig.suptitle(
            f"SeqVaeRawV4 raw diagnostics -- epoch {epoch}, sample {s} ({guid[:16]}) [{unit}]",
            fontsize=13,
        )

        # --- Row 0: predictive-band overlays at three anchors ---------------
        anchors = self._pick_anchors(t_valid, warmup)
        for col, t in enumerate(anchors):
            ax = axes[0][col]
            xr = torch.arange(horizon * r)
            true_hr = x_plus[s, t].reshape(-1)
            mu_hr = mu_full[s, t].reshape(-1)
            sig_hr = sigma_full[s, t].reshape(-1)
            mu_bpm = _np(_denorm_fhr(mu_hr, stats))
            true_bpm = _np(_denorm_fhr(true_hr, stats))
            lo = _np(_denorm_fhr(mu_hr - 2.0 * sig_hr, stats))
            hi = _np(_denorm_fhr(mu_hr + 2.0 * sig_hr, stats))
            ax.fill_between(_np(xr), lo, hi, color=COLOR_BLUE, alpha=0.20, label=r"$\mu\pm2\sigma$")
            ax.plot(_np(xr), mu_bpm, color=COLOR_BLUE, lw=1.4, label=r"$\hat x$ (full)")
            ax.plot(_np(xr), true_bpm, color=COLOR_VERMILLION, lw=1.0, ls="--", label=r"$x^+$ true")
            cover = float(((true_hr >= (mu_hr - 2 * sig_hr)) & (true_hr <= (mu_hr + 2 * sig_hr)))
                          .float().mean())
            ax.set_title(f"anchor t={t}  (2σ cover {cover:.0%})", fontsize=9)
            ax.set_xlabel("future raw sample")
            ax.set_ylabel(f"FHR ({unit})")
            if col == 0:
                ax.legend(fontsize=7, loc="best")

        # --- Row 1a: per-dim KL (G4) ---------------------------------------
        kl_dim = self._per_dim_kl(outs, s, warmup)
        ax = axes[1][0]
        ax.bar(range(kl_dim.shape[0]), _np(kl_dim), color=COLOR_ORANGE)
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_title("per-dim KL (mean over support)", fontsize=9)
        ax.set_xlabel(r"latent dim $d$")
        ax.set_ylabel("KL (nats)")

        # --- Row 1b: K_true vs K_shuffled per step (G6) --------------------
        ax = axes[1][1]
        k_true = _np(outs["kld_per_t"][s])
        ax.plot(k_true, color=COLOR_BLACK, lw=1.2, label=r"$K_{\mathrm{true}}$")
        if perm_out is not None and "kld_shuffled_per_t" in perm_out:
            k_shuf = _np(perm_out["kld_shuffled_per_t"][s])
            ax.plot(k_shuf, color=COLOR_GRAY, lw=1.0, ls="--", label=r"$K_{\mathrm{shuffled}}$")
        ax.axvspan(0, warmup, color=COLOR_GRAY, alpha=0.15)
        ax.set_title(r"$K_t$: true vs shuffled source", fontsize=9)
        ax.set_xlabel("anchor step t")
        ax.set_ylabel("KL (nats)")
        ax.legend(fontsize=7)

        # --- Row 1c: per-horizon raw forecast MAE --------------------------
        ax = axes[1][2]
        mae_h = self._mae_per_horizon(mu_full[s], x_plus[s], f_mask[s])
        ax.plot(range(1, horizon + 1), _np(mae_h), color=COLOR_BLUE, marker="o", ms=3)
        ax.set_title(f"raw forecast MAE per horizon ({unit})", fontsize=9)
        ax.set_xlabel(r"horizon step $\tau$")
        ax.set_ylabel(f"MAE ({unit})")

        # --- Row 2a: lag-attention matrix ----------------------------------
        ax = axes[2][0]
        attn = _np(outs["attn_weights"][s])
        while attn.ndim > 2:
            attn = attn.mean(axis=1)
        im = ax.imshow(attn, aspect="auto", origin="lower", cmap="magma")
        ax.set_title("lag attention (mean over heads)", fontsize=9)
        ax.set_xlabel("lag ℓ")
        ax.set_ylabel("anchor t")
        fig.colorbar(im, ax=ax, fraction=0.046)

        # --- Row 2b: TE-lag map --------------------------------------------
        ax = axes[2][1]
        te = _np(outs["te_lag_map"][s])
        while te.ndim > 2:
            te = te.mean(axis=0)
        im = ax.imshow(te, aspect="auto", origin="lower", cmap="viridis")
        ax.set_title("TE-lag map", fontsize=9)
        ax.set_xlabel("lag ℓ")
        ax.set_ylabel("anchor t")
        fig.colorbar(im, ax=ax, fraction=0.046)

        # --- Row 2c: pred_gap-vs-K pathology summary -----------------------
        ax = axes[2][2]
        ax.axis("off")
        k_raw = float(_np(outs["kld_per_t"][s])[warmup:].mean()) if t_valid > warmup else 0.0
        k_shuf = (
            float(_np(perm_out["kld_shuffled_per_t"][s])[warmup:].mean())
            if perm_out is not None and t_valid > warmup else float("nan")
        )
        mean_lv = float(outs["logvar_full"][s].mean())
        active = float(outs.get("kld_active_frac", torch.zeros(())))
        pathology = "OK" if not (k_raw > 1e-3) else "watch pred_gap>0"
        lines = [
            "pathology guard",
            f"  K_raw (mean)      = {k_raw:.4f}",
            f"  K_shuffled (mean) = {k_shuf:.4f}",
            f"  kld_active_frac   = {active:.3f}",
            f"  mean logvar_full  = {mean_lv:.3f}",
            "",
            "K >> 0 with pred_gap <= 0",
            "  => latent info not used",
            f"  status: {pathology}",
        ]
        ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", family="monospace",
                fontsize=9, transform=ax.transAxes)

        fig.tight_layout(rect=(0, 0, 1, 0.97))
        return fig

    # ------------------------------------------------------------------
    # Small numeric helpers
    # ------------------------------------------------------------------
    def _pick_anchors(self, t_valid: int, warmup: int):
        """Three anchors spread across the valid range, starting near ``forecast_anchor_frac``."""
        lo = max(warmup, 0)
        hi = max(lo + 1, t_valid - 1)
        base = int(self.forecast_anchor_frac * (hi - lo)) + lo
        cands = sorted({lo, min(base, hi), hi})
        while len(cands) < 3:
            cands.append(min(cands[-1] + 1, hi))
        return cands[:3]

    @staticmethod
    def _per_dim_kl(outs, s: int, warmup: int) -> torch.Tensor:
        """Closed-form diagonal-Gaussian KL per latent dim, averaged over the support steps."""
        mp = outs["mu_prior"][s]
        lp = outs["logvar_prior"][s]
        mq = outs["mu_post"][s]
        lq = outs["logvar_post"][s]
        kl = 0.5 * (lp - lq + (torch.exp(lq) + (mq - mp) ** 2) / torch.exp(lp) - 1.0)  # (T, d_z)
        if kl.size(0) > warmup:
            kl = kl[warmup:]
        return kl.mean(dim=0)

    @staticmethod
    def _mae_per_horizon(mu_tvhr, x_tvhr, m_tvhr) -> torch.Tensor:
        """Masked mean |μ − x⁺| per horizon step τ (averaged over anchors and raw substeps)."""
        x = torch.nan_to_num(x_tvhr, nan=0.0, posinf=0.0, neginf=0.0)
        err = (mu_tvhr - x).abs() * m_tvhr                 # (T_valid, H, R)
        num = err.sum(dim=(0, 2))                          # (H,)
        den = m_tvhr.sum(dim=(0, 2)).clamp_min(1.0)
        return num / den
