

class ComprehensiveForecastPlotCallback(Callback):
    """Comprehensive visualization of reconstructions and scattering forecasts."""

    def __init__(self, output_dir: Path | str, plot_every_epoch: int = 5, *, predictive_horizon: Optional[int] = None, num_examples: int = 1, max_anchors: int = 3) -> None:
        super().__init__()
        self.output_dir = Path(output_dir) / 'forecast_visualizations'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plot_every_epoch = max(1, int(plot_every_epoch))
        self.predictive_horizon = predictive_horizon
        self.num_examples = max(1, int(num_examples))
        self.max_anchors = max(1, int(max_anchors))

    @staticmethod
    def _first_batch(trainer):
        if hasattr(trainer, 'datamodule') and trainer.datamodule is not None:
            dataloader = trainer.datamodule.val_dataloader()
            if isinstance(dataloader, list):
                dataloader = dataloader[0]
        else:
            dataloader = trainer.val_dataloaders
            if isinstance(dataloader, list):
                dataloader = dataloader[0]
        try:
            return next(iter(dataloader))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"ComprehensiveForecastPlotCallback: could not fetch validation batch ({exc})")
            return None

    @staticmethod
    def _select_indices(total: int, count: int) -> list[int]:
        if total <= count:
            return list(range(total))
        step = max(1, total // count)
        return [idx for idx in range(0, total, step)][:count]

    def on_validation_epoch_end(self, trainer, pl_module) -> None:  # type: ignore[override]
        epoch = trainer.current_epoch
        if (epoch + 1) % self.plot_every_epoch != 0 or not trainer.is_global_zero:
            return
        batch = self._first_batch(trainer)
        if batch is None:
            return
        batch = pl_module.transfer_batch_to_device(batch, pl_module.device, dataloader_idx=0)
        model = pl_module.model
        orig_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        pl_module.eval()
        try:
            with torch.no_grad():
                outputs = orig_model(batch.fhr_st, batch.fhr_ph, batch.fhr_up_ph)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"ComprehensiveForecastPlotCallback: forward pass failed: {exc}")
            pl_module.train()
            return
        finally:
            pl_module.train()
        mu_pr = outputs.get('mu_pr')
        logvar_pr = outputs.get('logvar_pr')
        mu_post = outputs.get('mu_post')
        mu_prior = outputs.get('mu_prior')
        latent_z = outputs.get('z')

        self._plot_reconstruction_overview(
            y_raw=batch.fhr,
            up_raw=getattr(batch, 'up', None),
            mu_pr=mu_pr,
            logvar_pr=logvar_pr,
            latent_z=latent_z,
            mu_post=mu_post,
            mu_prior=mu_prior,
            epoch=epoch,
        )

        if getattr(orig_model, 'has_forecaster', lambda: False)():
            with torch.no_grad():
                forecast = orig_model.forecast_scattering(
                    y_st=batch.fhr_st,
                    y_ph=batch.fhr_ph,
                    x_ph=batch.fhr_up_ph,
                    timesteps=None,
                    use_posterior_mean=True,
                )
            mu_future = forecast.get('mu_future')
            logvar_future = forecast.get('logvar_future')
            timesteps = forecast.get('timesteps')
            target_stph = torch.cat([batch.fhr_st, batch.fhr_ph], dim=-1)
            self._plot_scattering_forecast(epoch, mu_future, logvar_future, timesteps, target_stph)
            self._plot_latent_summary(epoch, mu_post, mu_prior)

    def _plot_reconstruction_overview(self, y_raw: torch.Tensor, up_raw: Optional[torch.Tensor], mu_pr: Optional[torch.Tensor], logvar_pr: Optional[torch.Tensor], latent_z: Optional[torch.Tensor], mu_post: Optional[torch.Tensor], mu_prior: Optional[torch.Tensor], epoch: int) -> None:
        import matplotlib.pyplot as plt
        import numpy as np
        if mu_pr is None or logvar_pr is None or y_raw is None:
            return
        sample_idx = 0
        try:
            gt = y_raw[sample_idx].detach().cpu().numpy()
            recon = mu_pr[sample_idx].detach().cpu().numpy()
            logvar = logvar_pr[sample_idx].detach().cpu().numpy()
        except Exception:
            return
        std = np.sqrt(np.maximum(np.exp(logvar), 1e-8))
        diff = gt - recon
        up = None
        if up_raw is not None:
            try:
                up = up_raw[sample_idx].detach().cpu().numpy()
            except Exception:
                up = None
        latent = None
        if latent_z is not None:
            latent = latent_z[sample_idx].detach().cpu().numpy()
        elif mu_post is not None:
            latent = mu_post[sample_idx].detach().cpu().numpy()
        prior = None
        if mu_prior is not None:
            try:
                prior = mu_prior[sample_idx].detach().cpu().numpy()
            except Exception:
                prior = None
        Fs = 4.0
        t = np.arange(len(gt)) / Fs
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        mae = float(np.mean(np.abs(diff)))
        corr = float('nan')
        if np.std(gt) > 1e-8 and np.std(recon) > 1e-8:
            try:
                corr = float(np.corrcoef(gt, recon)[0, 1])
            except Exception:
                corr = float('nan')
        plt.style.use('default')
        fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
        axes[0].plot(t, gt, label='FHR', color='#055C9A', linewidth=1.2)
        if up is not None:
            axes[0].plot(t, up, label='UP', color='#0DD8A2', linewidth=1.0, alpha=0.8)
        axes[0].set_title('Raw signals (FHR/UP)')
        axes[0].grid(alpha=0.3)
        axes[0].legend(loc='upper right')
        axes[1].plot(t, gt, label='Ground truth', color='#456882', linewidth=1.2)
        axes[1].plot(t, recon, label='Reconstruction', color='#BB3E00', linewidth=1.0)
        axes[1].fill_between(t, recon - std, recon + std, color='#F7AD45', alpha=0.3, label='uncertainty band')
        axes[1].set_title(f'Reconstruction vs ground truth | RMSE={rmse:.3f} | MAE={mae:.3f} | Corr={corr:.3f}')
        axes[1].grid(alpha=0.3)
        axes[1].legend(loc='upper right')
        if latent is not None:
            im = axes[2].imshow(latent.T, aspect='auto', origin='lower', cmap='RdBu_r')
            axes[2].set_title('Latent representation (posterior)')
            axes[2].set_ylabel('Latent dim')
            axes[2].set_xlabel('Decimated step')
            plt.colorbar(im, ax=axes[2], orientation='vertical', pad=0.01)
        else:
            axes[2].text(0.5, 0.5, 'Latent trajectory unavailable', ha='center', va='center')
            axes[2].set_axis_off()
        fig.tight_layout()
        save_path = self.output_dir / f'reconstruction_overview_epoch_{epoch:04d}.pdf'
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
        logger.info(f"Saved reconstruction overview to {save_path}")

    def _plot_scattering_forecast(self, epoch: int, mu_future: Optional[torch.Tensor], logvar_future: Optional[torch.Tensor], timesteps: Optional[torch.Tensor], target_stph: torch.Tensor) -> None:
        import matplotlib.pyplot as plt
        import numpy as np
        if mu_future is None or timesteps is None or timesteps.numel() == 0:
            logger.warning('ComprehensiveForecastPlotCallback: no scattering predictions available for plotting.')
            return
        sample_idx = 0
        mu_np = mu_future[sample_idx].detach().cpu().numpy()
        anchors = timesteps.detach().cpu().numpy()
        selected_idx = self._select_indices(len(anchors), self.max_anchors)
        rows = len(selected_idx)
        if rows == 0:
            return
        fig, axes = plt.subplots(rows, 3, figsize=(15, 4 * rows), constrained_layout=True)
        if rows == 1:
            axes = np.expand_dims(axes, axis=0)
        mse_values = []
        stph_np = target_stph[sample_idx].detach().cpu().numpy()
        for row, anchor_idx in enumerate(selected_idx):
            anchor = int(anchors[anchor_idx])
            pred = mu_np[anchor_idx]
            horizon = pred.shape[0]
            target = stph_np[anchor + 1: anchor + 1 + horizon]
            if target.shape[0] != pred.shape[0]:
                min_len = min(target.shape[0], pred.shape[0])
                pred = pred[:min_len]
                target = target[:min_len]
            diff = pred - target
            mse = float(np.mean(diff ** 2)) if target.size else float('nan')
            mse_values.append((anchor, mse))
            im0 = axes[row, 0].imshow(target.T, aspect='auto', origin='lower', cmap='viridis')
            axes[row, 0].set_title(f'Target scattering | t={anchor}')
            plt.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.01)
            im1 = axes[row, 1].imshow(pred.T, aspect='auto', origin='lower', cmap='viridis')
            axes[row, 1].set_title('Predicted scattering')
            plt.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.01)
            im2 = axes[row, 2].imshow(np.abs(diff).T, aspect='auto', origin='lower', cmap='magma')
            axes[row, 2].set_title(f'|Error| (MSE={mse:.4f})')
            plt.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.01)
            for col in range(3):
                axes[row, col].set_ylabel('Channel index')
                axes[row, col].set_xlabel('Horizon step')
        fig.suptitle('Scattering forecast vs target', fontsize=14)
        save_path = self.output_dir / f'scattering_forecast_epoch_{epoch:04d}.pdf'
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
        if mse_values:
            summary = ', '.join(f"t={t}: MSE={m:.4f}" for t, m in mse_values)
            logger.info(f'Scattering forecast summary (epoch {epoch}): {summary}')

    def _plot_latent_summary(self, epoch: int, mu_post: Optional[torch.Tensor], mu_prior: Optional[torch.Tensor]) -> None:
        import matplotlib.pyplot as plt
        import numpy as np
        if mu_post is None or mu_prior is None:
            return
        sample_idx = 0
        post = mu_post[sample_idx].detach().cpu().numpy()
        prior = mu_prior[sample_idx].detach().cpu().numpy()
        diff = post - prior
        fig, axes = plt.subplots(1, 3, figsize=(18, 4), constrained_layout=True)
        im0 = axes[0].imshow(prior.T, aspect='auto', origin='lower', cmap='RdBu_r')
        axes[0].set_title('Prior mean')
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.01)
        im1 = axes[1].imshow(post.T, aspect='auto', origin='lower', cmap='RdBu_r')
        axes[1].set_title('Posterior mean')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.01)
        im2 = axes[2].imshow(diff.T, aspect='auto', origin='lower', cmap='coolwarm')
        axes[2].set_title('Posterior - Prior')
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.01)
        for ax in axes:
            ax.set_xlabel('Decimated step')
            ax.set_ylabel('Latent dim')
        fig.suptitle(f'Latent summary - epoch {epoch}', fontsize=14)
        save_path = self.output_dir / f'latent_summary_epoch_{epoch:04d}.pdf'
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
