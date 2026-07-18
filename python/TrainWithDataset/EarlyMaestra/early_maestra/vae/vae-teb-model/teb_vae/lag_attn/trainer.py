r"""The experiment driver: build the model from config, build the callbacks, run the fit.

Run from the repository root, which is what puts ``teb_vae``, ``train`` and ``utils`` on
``sys.path``:

.. code-block:: bash

    # Single GPU / local smoke
    python -m teb_vae.lag_attn.trainer --config teb_vae/lag_attn/configs/tiny.yaml

    # Prod box. The rank count must equal len(general_config.cuda_devices) -- default.yaml ships 7 --
    # or Lightning rejects the device/world-size mismatch at Trainer construction. Export the
    # stamp so ranks 1..N-1 share rank 0's run directory.
    TEB_RUN_STAMP="$(date '+%Y-%m-%d--[%H-%M]')" torchrun --nproc_per_node=7 \
        -m teb_vae.lag_attn.trainer --config teb_vae/lag_attn/configs/default.yaml

Everything about *running* an experiment -- run directories, seeding, log sinks, MLflow, the
``Trainer`` itself -- is the framework's. This module supplies the three things the framework
cannot know: how to turn ``model_config.VAE_model`` into a net, which callbacks this model wants,
and which DDP strategy its parameter-usage pattern permits.
"""
from __future__ import annotations

import argparse
import inspect
import tempfile
import time
from typing import Any, Dict, List

import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from loguru import logger

from teb_vae.lag_attn.config import resolve_config_file
from teb_vae.lag_attn.nets.model import SeqVaeLagAttn
from teb_vae.lag_attn.task import SeqVaeLagAttnTask
from train.callbacks import (
    HyperparameterLoggingCallback,
    LossPlotCallback,
    MetricsHistoryCsvCallback,
    MetricsLoggingCallback,
)
from train.data_module import GraphDataModule
from train.graph_model_base import GraphModelBase
from train.graph_models_utils import check_model_class, load_checkpoint_strict

#: Every metric suffix the task emits. Kept here rather than imported from the task because it is
#: the trainer that must tell ``MetricsLoggingCallback`` what to collect -- that callback hardcodes
#: its own default list, which does not match this model. A test drives the task and fails if this
#: list ever falls behind what it emits.
_METRIC_SUFFIXES = (
    "total_loss", "main_loss", "feat_loss", "base_loss",
    "kld_loss", "kld_raw", "kld_train", "kld_active_frac",
    "perm_loss", "kld_shuffled", "kld_shuffled_ratio",
    "feat_loss_shuffled", "shuffle_penalty",
    "mean_logvar_full", "mean_logvar_base",
    "pred_gap", "delta_mu_rms", "mu_post_prior_gap_rms",
    "kld_beta", "lambda_full", "lambda_base",
    "lag_smoothness",
    # The tanh-bounded latent heads' saturation fractions: the tell for a head pinned at its
    # bound, where its gradient vanishes and the latent stops responding to the source.
    "mu_prior_sat_frac", "delta_mu_sat_frac",
)

#: Metric suffixes the *framework* injects rather than the task, and only on training batches when
#: the spike breaker is enabled (``pl_model_base.py:487-489``). They are the whole diagnostic
#: surface of the breaker -- the per-step skip decision and the EMA it is comparing against -- and
#: this repository has already lost days to a run that trained normally and then skipped every
#: batch forever, which is a failure only these two columns can show. Train-only: the breaker never
#: runs on a validation batch, so a ``val/`` variant would be NaN in every row of every run.
_TRAIN_ONLY_SUFFIXES = ("spike_skipped", "spike_ema_loss")

#: The names the framework actually puts in ``callback_metrics``: every task metric is logged as
#: ``{stage}/{name}``, plus the bare ``lr`` the base logs once per epoch. A bare suffix here would
#: match nothing and produce a column that is NaN for every epoch of every run.
_TRACKED_METRICS = (
    tuple(f"{stage}/{name}" for stage in ("train", "val") for name in _METRIC_SUFFIXES)
    + tuple(f"train/{name}" for name in _TRAIN_ONLY_SUFFIXES)
    + ("lr",)
)

#: ``VAE_model`` keys that name a nested block rather than a constructor argument, and the one key
#: that must never be forwarded (weight initialisation is not a config decision).
_NON_CONSTRUCTOR_KEYS = frozenset({"horizon_refine", "encoder", "init_weights"})


class LagAttnTrainer(GraphModelBase):
    """Experiment driver for :class:`~teb_vae.lag_attn.nets.model.SeqVaeLagAttn`."""

    def _build_model_kwargs(self) -> Dict[str, Any]:
        """Translate the ``model_config.VAE_model`` block into constructor kwargs.

        One ``inspect.signature`` sweep: any flat key naming a real constructor argument is
        forwarded, and anything else -- the loss weights, the likelihood, the schedule -- is left
        for the task. Deliberately no table of defaults: the constructor already owns those, and a
        second copy here could only ever disagree with it.

        The two nested blocks and ``logvar_clamp`` are the only translations, because they are the
        only places where the config's shape and the constructor's differ.

        Returns:
            Constructor kwargs for the net.
        """
        vae_config = (self.config.get("model_config", {}) or {}).get("VAE_model", {}) or {}
        valid_parameters = set(inspect.signature(SeqVaeLagAttn.__init__).parameters)

        kwargs: Dict[str, Any] = {
            name: value
            for name, value in vae_config.items()
            if name in valid_parameters and name not in _NON_CONSTRUCTOR_KEYS and value is not None
        }

        # Nested blocks: grouped in the config for readability, flat on the constructor.
        horizon_config = vae_config.get("horizon_refine") or {}
        for config_key, kwarg in (("depth", "horizon_depth"), ("kernel", "horizon_kernel"), ("film", "horizon_film")):
            if config_key in horizon_config:
                kwargs[kwarg] = horizon_config[config_key]
        encoder_config = vae_config.get("encoder") or {}
        if "extra_dilations" in encoder_config:
            kwargs["encoder_extra_dilations"] = tuple(
                int(dilation) for dilation in (encoder_config["extra_dilations"] or [])
            )

        # YAML has no tuple; the constructor's bound is a pair.
        logvar_clamp = vae_config.get("logvar_clamp")
        if isinstance(logvar_clamp, (list, tuple)) and len(logvar_clamp) == 2:
            kwargs["logvar_clamp"] = (float(logvar_clamp[0]), float(logvar_clamp[1]))

        return kwargs

    def create_model(self) -> None:
        """Build the net, optionally load a checkpoint into it, and wrap it in its task.

        Raises:
            RuntimeError: If ``core_model_checkpoint`` is set but cannot be aligned into the model.
                ``load_checkpoint_strict`` returns ``None`` rather than raising when nothing lines
                up, so an unchecked call trains a randomly-initialised model that was supposed to
                be warm-started -- and reports nothing.
        """
        model_kwargs = self._build_model_kwargs()
        logger.info(
            "Building SeqVaeLagAttn with kwargs: "
            + ", ".join(f"{key}={value}" for key, value in model_kwargs.items())
        )
        self.pytorch_model = SeqVaeLagAttn(**model_kwargs)

        if not self.pytorch_model.causal_norm:
            logger.warning(
                "causal_norm=False: the encoders' GroupNorm pools statistics across time, so the "
                "prior conditions on the future and kld_raw is NOT a transfer-entropy surrogate."
            )

        model_config = self.config.get("model_config", {}) or {}
        vae_config = model_config.get("VAE_model", {}) or {}

        self.checkpoint = model_config.get("core_model_checkpoint")
        if self.checkpoint is not None:
            blob = torch.load(str(self.checkpoint), map_location="cpu", weights_only=False)
            # Before the load, not after: a blob from another model may align by accident.
            check_model_class(blob, SeqVaeLagAttn.__name__)
            if load_checkpoint_strict(model=self.pytorch_model, checkpoint=blob) is None:
                raise RuntimeError(
                    f"could not align core_model_checkpoint {self.checkpoint!r} into "
                    f"SeqVaeLagAttn (no matching module keys). Training would otherwise continue "
                    f"from random weights."
                )
            logger.info(f"Model loaded from checkpoint: {self.checkpoint}")

        self.pl_model = SeqVaeLagAttnTask(
            self.pytorch_model,
            lr=self.lr,
            lr_milestones=self.lr_milestones,
            model_kwargs=model_kwargs,
            spike_breaker=self.config.get("advanced_config", {}).get("spike_breaker"),
            beta_schedule=vae_config.get("beta_schedule"),
            kld_beta=vae_config.get("kld_beta", 0.01),
            lambda_full=vae_config.get("lambda_full", 1.0),
            lambda_base=vae_config.get("lambda_base", 0.5),
            likelihood=vae_config.get("likelihood", "gaussian_nll"),
            sigma_obs=vae_config.get("sigma_obs", "learned"),
            free_bits=vae_config.get("free_bits", 0.0),
            detach_baseline_in_full=vae_config.get("detach_baseline_in_full", False),
            lambda_lag=vae_config.get("lag_smoothness_lambda", 0.0),
        )
        # Re-forces the config's values onto hparams, so a checkpoint-restored run follows the
        # config it was launched with rather than the one it was originally trained under.
        self.apply_config_hyperparameters(
            {"lr": self.lr, "lr_milestones": self.lr_milestones}, self.pl_model
        )

    def select_ddp_strategy(self, num_devices: int, config: Dict[str, Any], model=None) -> str:
        r"""Select the Lightning ``strategy`` string from the configured parameter usage.

        Plain ``'ddp'`` implies ``find_unused_parameters=False``, under which the reducer expects
        **every** parameter to be marked ready in every backward. Two groups of parameters can go
        unused here, and both are decided by config:

        * The decoder log-variance heads are consumed exactly when ``likelihood='gaussian_nll'``
          and ``sigma_obs='learned'``. A fixed ``sigma_obs`` starves them.
        * With ``head_structured_latent=True`` the posterior consumes the per-head summaries, so
          the attention's output projection feeds only a diagnostic key and never receives a
          gradient -- unless ``freeze_unused_attn_proj`` has taken it out of the reducer's
          expectation set.

        Everything is read from ``config`` and the ``model`` argument goes unused, which is
        deliberate rather than lazy. The framework passes the *Lightning module* here, not the raw
        net, so reaching for a model attribute like ``frozen_attn_proj`` would find nothing,
        silently conclude the projection was starved, and regress a correctly-configured run to the
        slower strategy with no error.

        Args:
            num_devices: Number of CUDA devices for the run.
            config: The resolved config.
            model: The Lightning module, unused. See above.

        Returns:
            The Lightning ``strategy`` string.
        """
        if num_devices <= 1:
            return "auto"
        vae_config = (config.get("model_config", {}) or {}).get("VAE_model", {}) or {}
        sigma_obs = vae_config.get("sigma_obs", "learned")
        logvar_heads_consumed = (
            str(vae_config.get("likelihood", "gaussian_nll")) == "gaussian_nll"
            and isinstance(sigma_obs, str)
            and sigma_obs == "learned"
        )
        head_structured_latent = bool(vae_config.get("head_structured_latent", False))
        # The same expression the net uses to decide whether to freeze the projection.
        frozen_attn_proj = bool(vae_config.get("freeze_unused_attn_proj", False)) and head_structured_latent
        attn_proj_starved = head_structured_latent and not frozen_attn_proj

        if logvar_heads_consumed and not attn_proj_starved:
            return "ddp"
        return "ddp_find_unused_parameters_true"

    def train_model(self, train_loader, validation_loader):
        """Build this model's callbacks and run the fit.

        Args:
            train_loader: Training dataloader.
            validation_loader: Validation dataloader.

        Returns:
            The fitted Lightning ``Trainer``.
        """
        callbacks_config = self.config.get("advanced_config", {}).get("callbacks", {}) or {}
        checkpoint_config = callbacks_config.get("model_checkpoint", {}) or {}

        self.metrics_callback = MetricsLoggingCallback(tracked_metrics=_TRACKED_METRICS)
        self.metrics_csv_callback = MetricsHistoryCsvCallback(
            source=self.metrics_callback, output_dir=self.train_results_dir
        )
        self.loss_plot_callback = LossPlotCallback(
            output_dir=self.train_results_dir,
            plot_frequency=self.plot_every_epoch,
            mlflow_logger=self.mlflow_logger,
        )
        self.hyperparam_callback = HyperparameterLoggingCallback(
            # Explicit, because the default list names keys this model never emits: it asks for a
            # bare `kld_beta`, which the framework renames to `train/kld_beta` on the way out, and
            # for `hyperparams/beta`, which nothing logs at all. Left to default, every series is
            # NaN and the beta ramp -- the one knob the config tells the operator to retune against
            # the kld_raw trajectory -- is silently missing from hyperparameters.html.
            tracked_keys=("train/kld_beta", "lr"),
            output_dir=self.train_results_dir,
            plot_frequency=10,
            mlflow_logger=self.mlflow_logger,
        )
        self.checkpoint_callback = ModelCheckpoint(
            dirpath=self.model_checkpoint_dir,
            # From config: hardcoding it here would make the config key a decoration.
            monitor=checkpoint_config.get("monitor", "val/total_loss"),
            # Lightning auto-prefixes each placeholder with its own name, so "model-epoch={epoch}"
            # would render as "model-epoch=epoch=00".
            filename="lag-attn-{epoch:02d}",
            save_top_k=checkpoint_config.get("save_top_k", 3),
            mode=checkpoint_config.get("mode", "min"),
        )
        callback_list: List[Callback] = [
            self.metrics_callback,
            self.metrics_csv_callback,
            self.loss_plot_callback,
            self.hyperparam_callback,
            self.checkpoint_callback,
        ]

        # The diagnostic plotter is opt-in and pulls matplotlib, so it is imported only when the
        # config asks for it -- a smoke run that disables it pays neither the import nor the
        # per-epoch figure cost.
        plot_config = callbacks_config.get("lag_attn_plotting", {}) or {}
        if plot_config.get("enabled", False):
            from teb_vae.lag_attn.plotting import LagAttnPlotCallback

            callback_list.append(
                LagAttnPlotCallback(
                    output_dir=self.train_results_dir,
                    plot_frequency=plot_config.get("plot_frequency", 1),
                    num_examples=plot_config.get("num_examples", 2),
                    file_format=plot_config.get("file_format", "pdf"),
                    forecast_channels=plot_config.get("forecast_channels", (0, 43, 80)),
                    forecast_anchor_frac=plot_config.get("forecast_anchor_frac", 0.6),
                    mlflow_logger=self.mlflow_logger,
                )
            )

        trainer = self.build_trainer(callback_list, model=self.pl_model)
        trainer.fit(self.pl_model, train_loader, validation_loader)
        return trainer


def main(config_path: str) -> None:
    """Resolve the config, build everything, and run the fit.

    The call order is load-bearing and nothing chains it. ``setup_config`` is what seeds the run,
    creates the output directories, opens the log sinks and connects MLflow; building the model
    before it means no seeding, no logs, nowhere to write, and ``mlflow_logger is None`` -- which
    silently drops the MLflow callback from the fit.

    Args:
        config_path: Path to the YAML config. Its ``base:`` chain is resolved first.
    """
    start_time = time.time()

    # Resolved to a file because the driver reads a path and takes no dict, and into a temporary
    # directory because the run directory does not exist until setup_config, which needs the file
    # first. Deleted on the way out: the driver reads it once, in its constructor, and the durable
    # record of what ran is the framework's own -- the config dump in the run log and the
    # resolved_config.yaml MLflow artifact, both written from memory. Left behind, every rank of
    # every launch would strand one.
    with tempfile.TemporaryDirectory(prefix="teb_vae_") as resolved_dir:
        resolved_path = resolve_config_file(config_path, resolved_dir)
        logger.info(f"resolved config {config_path} -> {resolved_path}")

        graph_model = LagAttnTrainer(config_file_path=resolved_path)
        graph_model.setup_config()

    dataset_config = graph_model.config.get("dataset_config", {}) or {}
    if dataset_config.get("stat_path") is None:
        # The loader passes None straight through and the dataset merely skips normalization, so an
        # absent or misspelled stat_path (the config key is `stat_path`; the loader's parameter is
        # `stats_path`) would otherwise train a model on raw-scale inputs and report nothing.
        raise ValueError(
            "dataset_config.stat_path must be set; without it normalization is silently disabled "
            "and the model trains on unnormalized inputs."
        )
    data_module = GraphDataModule(graph_model.config)

    graph_model.create_model()
    graph_model.train_model(data_module.train_dataloader(), data_module.val_dataloader())
    logger.info(f"Training completed in {(time.time() - start_time) / 60:.2f} minutes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML config, e.g. teb_vae/lag_attn/configs/default.yaml. Run from the repo root.",
    )
    main(parser.parse_args().config)
