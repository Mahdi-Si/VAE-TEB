r"""Lightning wrapper + Graph-model trainer for :class:`SeqVaeRawV4` (raw-signal VAE-TEB v4).

Subclasses the v3 training stack (:class:`SeqVaeLagAttnV3Pl` /
:class:`GraphModelVaeTebLagAttnV3Trainer`), inheriting every scientific-cleanliness mechanism
unchanged -- the ``torch.compile`` bypass, the $\beta$ schedule, the loss-spike circuit breaker
(watched on the perm-free ``main_loss``), the fused source-permutation control, the honest
KL reporting, and the plain-``'ddp'`` strategy selection. Only the two places that are
feature-domain in v3 are re-implemented for raw batches:

* :meth:`SeqVaeRawV4Pl.compute_loss_and_metrics` -- builds ``(raw, mask)`` from ``batch.fhr`` /
  ``batch.up`` / ``batch.weight`` (v3 reads ``batch.fhr_st`` / ``batch.fhr_ph``), calls the raw
  ``forward`` / ``compute_loss`` (positional ``(forward_outputs, fhr_raw, mask)``, not the
  feature-domain ``(forward_outputs, y_st, y_ph, weight)``), and reproduces the entire v3 metrics
  surface plus the three raw metrics (``lowpass_loss`` / ``smooth_loss`` / ``raw_mae``).
* :meth:`GraphModelVaeTebRawV4Trainer._build_model_kwargs` -- re-run against
  ``inspect.signature(SeqVaeRawV4.__init__)`` so the nested ``frontend`` block and ``raw_len`` /
  ``decimation`` reach the constructor (the inherited sweep is bound to v3's signature and would
  silently drop them), and :meth:`~GraphModelVaeTebRawV4Trainer.train_model` -- register the raw
  plotting callback + a raw checkpoint filename (v3 hardcodes the feature-domain plot callback).

:func:`warm_start_from_v3` best-effort aligns the shared v3 submodules (encoders / prior /
lag-attention / posterior / horizon core + latent buffers) into a fresh raw model, leaving the
front ends $F_y, F_u$ and the raw decoder heads at their initialisation.

Usage:
    python -m model.vae_teb_prediction.model.model_raw.trainer_raw_v4 \\
        --config model/vae_teb_prediction/model/model_raw/config_raw_v4.yaml
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Tuple

import lightning as pl
import numpy as np
import torch
import yaml
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.profilers import SimpleProfiler
from loguru import logger

from hdf5_dataset.hdf5_dataset import create_optimized_dataloader
from model.vae_teb_prediction.model.model_raw.raw_masks import kl_mask, low_rate_mask
from model.vae_teb_prediction.model.model_raw.vae_teb_raw_v4 import SeqVaeRawV4
from model.vae_teb_prediction.model.trainer_lag_attn_v3 import (
    GraphModelVaeTebLagAttnV3Trainer,
    MetricsHistoryCsvCallback,
    SeqVaeLagAttnV3Pl,
    _V3_METRIC_SUFFIXES,
)
from train.callbacks import (
    HyperparameterLoggingCallback,
    LossPlotCallback,
    MetricsLoggingCallback,
)
from train.graph_models_utils import _prepare_checkpoint_state_dict

#: The three raw-domain metric suffixes v4 adds on top of the v3 surface.
_V4_EXTRA_SUFFIXES: Tuple[str, ...] = ("lowpass_loss", "smooth_loss", "raw_mae")

#: All metric suffixes :meth:`SeqVaeRawV4Pl.compute_loss_and_metrics` logs (v3 + raw).
_V4_METRIC_SUFFIXES: Tuple[str, ...] = _V3_METRIC_SUFFIXES + _V4_EXTRA_SUFFIXES

#: The stage-prefixed names ``MetricsLoggingCallback`` must be told about explicitly.
_TRACKED_METRICS_V4: Tuple[str, ...] = tuple(
    f"{stage}/{name}" for stage in ("train", "val") for name in _V4_METRIC_SUFFIXES
) + ("lr", "kld_beta")

#: Parameters a fresh raw v4 model owns that a v3 checkpoint cannot supply. The raw decoders reuse
#: the v3 attribute names ``baseline_decoder`` / ``residual_decoder`` but carry differently-shaped
#: heads, so only their (shape-mismatched) head keys land in ``missing`` -- the shared ``.core``
#: submodule (``horizon_core``) aligns by name and shape and is warm-started.
_EXPECTED_MISSING_PREFIXES_V3: Tuple[str, ...] = (
    "frontend_y.",
    "frontend_u.",
    "baseline_decoder.",
    "residual_decoder.",
)
#: Parameters a v3 checkpoint carries that a raw v4 model has deleted (the feature adapters) or
#: reshaped (the feature decoder heads); reported as skipped, never loaded.
_EXPECTED_SKIPPED_PREFIXES_V3: Tuple[str, ...] = (
    "target_adapter.",
    "source_adapter.",
    "baseline_decoder.",
    "residual_decoder.",
)


# =============================================================================
# Lightning wrapper
# =============================================================================
class SeqVaeRawV4Pl(SeqVaeLagAttnV3Pl):
    r"""Lightning wrapper for :class:`SeqVaeRawV4`.

    Inherits the v3 ``training_step`` (spike breaker on the perm-free ``main_loss``),
    ``on_save_checkpoint`` (stamps ``model_class='SeqVaeRawV4'`` + ``model_kwargs``), the
    ``torch.compile`` bypass, the $\beta$ schedule, and the perm scheduler. Only
    :meth:`compute_loss_and_metrics` is re-implemented: raw batches have no ``fhr_st`` / ``fhr_ph``
    fields, and the raw ``compute_loss`` takes ``(forward_outputs, fhr_raw, mask)`` positionally.
    """

    def compute_loss_and_metrics(
        self, batch: Any, batch_idx: int, stage: str
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        r"""Raw forward pass, single-phase raw loss, and the unified metrics dict.

        Mirrors :meth:`SeqVaeLagAttnV3Pl.compute_loss_and_metrics` key-for-key (so the inherited
        ``training_step`` circuit breaker, the metrics CSV, and MLflow all work unchanged) but on
        the raw pathway. The per-anchor weight handed to the KL / permutation / residual-diagnostic
        machinery is the low-rate KL mask :math:`m^{\mathrm{KL}}_t = \mathbb 1[t\ge w]\,
        m^{\mathrm{low}}_t` (the raw analogue of v3's decimated ``batch.weight``), so
        ``kld_shuffled`` stays directly comparable to ``kld_raw``.

        Args:
            batch: A raw batch exposing ``fhr`` / ``up`` :math:`(B, L_{\mathrm{raw}})` and the
                decimated ``weight`` :math:`(B, \tilde T)`.
            batch_idx: Rank-invariant step index (gates the perm-control schedule).
            stage: ``'train'`` / ``'val'`` / ``'test'``.

        Returns:
            ``(train_loss, metrics)`` -- ``train_loss = main_loss + lambda_perm * L_perm`` on
            scheduled train steps, else ``main_loss``; ``metrics['main_loss']`` always carries the
            perm-free value the breaker watches.
        """
        model = self.orig_model
        fhr_raw, up_raw, mask = model._default_batch_to_inputs(batch)

        forward_outputs = self.model(fhr_raw, up_raw, mask)

        # Low-rate per-anchor validity weight (the raw analogue of v3's decimated batch.weight);
        # used for the KL/perm control and the residual diagnostics so every KL-space number is
        # weighted the same way ``compute_loss`` weights its own KL term.
        geo = model.geometry
        kl_weight = kl_mask(low_rate_mask(mask, geo), geo)

        beta = self._resolve_beta(self.current_epoch)
        lambda_full = float(self.hparams.get("lambda_full", 1.0))
        lambda_base = float(self.hparams.get("lambda_base", 0.5))
        likelihood = str(self.hparams.get("likelihood", "gaussian_nll"))
        sigma_obs = self.hparams.get("sigma_obs", "learned")
        if not isinstance(sigma_obs, str):
            sigma_obs = float(sigma_obs)
        free_bits = float(self.hparams.get("free_bits", 0.0))
        detach_baseline_in_full = bool(self.hparams.get("detach_baseline_in_full", True))
        lambda_lag = float(self.hparams.get("lambda_lag", 0.0))
        lambda_lp = float(self.hparams.get("lambda_lp", 0.5))
        lambda_smooth = float(self.hparams.get("lambda_smooth", 0.1))
        lowpass_scales = self.hparams.get("lowpass_scales", (4, 16, 32, 60))

        loss_dict = model.compute_loss(
            forward_outputs=forward_outputs,
            fhr_raw=fhr_raw,
            mask=mask,
            beta=beta,
            lambda_full=lambda_full,
            lambda_base=lambda_base,
            likelihood=likelihood,
            sigma_obs=sigma_obs,
            free_bits=free_bits,
            detach_baseline_in_full=detach_baseline_in_full,
            lambda_lp=lambda_lp,
            lambda_smooth=lambda_smooth,
            lowpass_scales=lowpass_scales,
            lambda_lag=lambda_lag,
        )
        main_loss = loss_dict["total_loss"]

        diag = self._compute_residual_diagnostics(
            forward_outputs=forward_outputs, weight=kl_weight
        )
        kld_raw = loss_dict["kld_raw"]

        metrics: Dict[str, Any] = {
            "total_loss": main_loss,
            "main_loss": main_loss.detach(),
            "feat_loss": loss_dict["feat_loss"],
            "base_loss": loss_dict["base_loss"],
            "kld_loss": loss_dict["kld_loss"],
            # G4 -- only kld_raw may be read as a TE surrogate.
            "kld_raw": kld_raw,
            "kld_train": loss_dict["kld_train"],
            "kld_active_frac": loss_dict["kld_active_frac"],
            "kld_beta": beta,
            "lambda_full": lambda_full,
            "lambda_base": lambda_base,
            # G7 -- variance-collapse early warning (sigma^2 -> exp(-5)).
            "mean_logvar_full": loss_dict["mean_logvar_full"],
            "mean_logvar_base": loss_dict["mean_logvar_base"],
            "mu_prior_sat_frac": forward_outputs["mu_prior_sat_frac"],
            "delta_mu_sat_frac": forward_outputs["delta_mu_sat_frac"],
            "delta_mu_rms": diag["delta_mu_rms"],
            "mu_post_prior_gap_rms": diag["mu_post_prior_gap_rms"],
            "pred_gap": loss_dict["base_loss"] - loss_dict["feat_loss"],
            "lag_smoothness": loss_dict["lag_smoothness"],
            # v4-specific raw-domain reporting.
            "lowpass_loss": loss_dict["lowpass_loss"],
            "smooth_loss": loss_dict["smooth_loss"],
            "raw_mae": loss_dict["raw_mae"],
        }

        # --- G6: fused source-permutation control (inherited, domain-agnostic) ---
        lambda_perm = float(model.lambda_perm)
        do_perm = self._sync_perm_decision(
            self._should_run_perm(batch_idx, fhr_raw.size(0), stage), fhr_raw.device
        )
        train_loss = main_loss
        if do_perm:
            optimise_perm = stage == "train" and lambda_perm > 0.0
            with torch.set_grad_enabled(optimise_perm and torch.is_grad_enabled()):
                perm = model.perm_kl_from_forward(
                    forward_outputs, weight=kl_weight, generator=self._perm_generator
                )
            kld_shuffled = perm["kld_shuffled"]
            if optimise_perm:
                train_loss = main_loss + lambda_perm * perm["perm_kl"]
                metrics["perm_loss"] = (lambda_perm * perm["perm_kl"]).detach()
            else:
                metrics["perm_loss"] = torch.zeros_like(kld_shuffled)
            metrics["kld_shuffled"] = kld_shuffled
            metrics["kld_shuffled_ratio"] = kld_shuffled / kld_raw.detach().clamp_min(1e-8)
            metrics["total_loss"] = train_loss

            # Prediction-space control (§12): a wrong source must be worse than no source,
            #     feat_loss < base_loss < feat_loss_shuffled.
            # The auxiliary low-pass/smooth/lag terms are irrelevant here (only feat_loss is read),
            # so they are switched off to keep the readout cheap.
            with torch.no_grad():
                permuted = model.perm_forward_outputs(
                    forward_outputs, perm_index=perm["perm_index"]
                )
                shuffled = model.compute_loss(
                    forward_outputs=permuted,
                    fhr_raw=fhr_raw,
                    mask=mask,
                    compute_kld_loss=False,
                    beta=0.0,
                    lambda_full=lambda_full,
                    lambda_base=lambda_base,
                    likelihood=likelihood,
                    sigma_obs=sigma_obs,
                    detach_baseline_in_full=detach_baseline_in_full,
                    lambda_lp=0.0,
                    lambda_smooth=0.0,
                    lambda_lag=0.0,
                )
            metrics["feat_loss_shuffled"] = shuffled["feat_loss"]
            metrics["shuffle_penalty"] = shuffled["feat_loss"] - loss_dict["feat_loss"]
        else:
            zero = torch.zeros_like(kld_raw)
            for key in ("perm_loss", "kld_shuffled", "kld_shuffled_ratio",
                        "feat_loss_shuffled", "shuffle_penalty"):
                metrics[key] = zero

        return train_loss, metrics


# =============================================================================
# Warm-start
# =============================================================================
def warm_start_from_v3(
    model: SeqVaeRawV4, checkpoint_path: str, *, strict_expectations: bool = True
) -> Dict[str, List[str]]:
    r"""Load a v3 checkpoint's shared submodules into a fresh raw v4 model.

    A raw v4 model shares the v3 encoders, prior, lag-attention, posterior, horizon core, and
    latent-stats buffers (all name- and shape-identical), but replaces the two feature adapters with
    the front ends $F_y, F_u$ and reshapes the two decoders into raw heads. A plain
    ``load_state_dict(strict=False)`` cannot be used directly: the raw decoders reuse the attribute
    names ``baseline_decoder`` / ``residual_decoder`` with differently-shaped head tensors, so PyTorch
    would raise a *shape-mismatch* ``RuntimeError`` (not a soft missing/unexpected). We therefore
    **filter** the v3 state to keys that exist in the v4 model with a matching shape, load that
    subset, and classify the rest.

    Note:
        The zero-init KL invariant survives warm-start only from a *freshly constructed* v3, whose
        ``posterior_head.delta_mu_head`` is zero. Warm-starting from a genuinely trained v3 loads a
        nonzero ``delta_mu_head`` (so :math:`\mu_q \neq \mu_p` and :math:`K > 0` at step 0) -- that
        is expected, not a bug. The raw residual decoder mean head is never loaded (it is new), so
        :math:`\Delta\hat X^{\mathrm{src}} = 0` at init regardless.

    Args:
        model: The freshly constructed :class:`SeqVaeRawV4`, mutated in place.
        checkpoint_path: Path to a :class:`SeqVaeLagAttnV3` checkpoint (Lightning ``.ckpt`` or a
            ``{model_class, model_state_dict, model_kwargs}`` blob).
        strict_expectations: If True, raise when the missing/skipped key sets contain anything beyond
            the known v3-vs-raw-v4 architectural delta (front ends + raw decoder heads vs. feature
            adapters + feature decoder heads).

    Returns:
        ``{"loaded": [...], "missing": [...], "skipped": [...]}`` parameter names -- ``loaded`` are the
        warm-started shared tensors, ``missing`` the v4-only tensors left at init, ``skipped`` the
        v3-only tensors not transferred.

    Raises:
        ValueError: If the checkpoint declares a ``model_class`` other than ``SeqVaeLagAttnV3``, if
            its state dict cannot be extracted, if zero tensors align, or if ``strict_expectations``
            is set and an unexplained key appears.
    """
    blob = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

    # NOT check_model_class: that guard asserts the blob matches the *active* class and would reject
    # the very cross-version load this function performs.
    declared = blob.get("model_class") if isinstance(blob, dict) else None
    if declared is not None and str(declared) != "SeqVaeLagAttnV3":
        raise ValueError(
            f"warm_start_from_v3 expects a SeqVaeLagAttnV3 checkpoint, got model_class="
            f"{declared!r}. For a same-class raw reload use model_config.core_model_checkpoint."
        )

    state = _prepare_checkpoint_state_dict(blob, map_location="cpu")
    if state is None:
        raise ValueError(f"could not extract a state dict from {checkpoint_path!r}")

    model_state = model.state_dict()
    filtered = {
        k: v
        for k, v in state.items()
        if k in model_state and tuple(model_state[k].shape) == tuple(v.shape)
    }
    model.load_state_dict(filtered, strict=False)

    loaded = sorted(filtered.keys())
    missing = sorted(set(model_state) - set(filtered))      # v4-only (front ends, raw heads)
    skipped = sorted(set(state) - set(filtered))            # v3-only (adapters, reshaped heads)

    def _explained(key: str, prefixes: Tuple[str, ...]) -> bool:
        return any(key.startswith(p) for p in prefixes)

    surprising_missing = [
        k for k in missing if not _explained(k, _EXPECTED_MISSING_PREFIXES_V3)
    ]
    surprising_skipped = [
        k for k in skipped if not _explained(k, _EXPECTED_SKIPPED_PREFIXES_V3)
    ]

    logger.info(
        "warm-start from {}: aligned {} shared tensors, left {} v4-only tensors at init, "
        "skipped {} v3-only tensors",
        checkpoint_path, len(loaded), len(missing), len(skipped),
    )
    if missing:
        logger.info("  v4-only (not warm-started): {}", ", ".join(missing))
    if skipped:
        logger.info("  v3-only (skipped): {}", ", ".join(skipped))

    if not loaded:
        raise ValueError(
            f"warm-start aligned ZERO tensors from {checkpoint_path!r}; the checkpoint's keys do "
            "not match this architecture. Check the v3 model_kwargs mirror the raw run's core "
            "(d_model, d_z, horizon_refine, encoder.extra_dilations, ...)."
        )
    if strict_expectations and (surprising_missing or surprising_skipped):
        raise ValueError(
            "warm-start produced key differences beyond the known v3-vs-raw-v4 delta.\n"
            f"  unexplained missing: {surprising_missing}\n"
            f"  unexplained skipped: {surprising_skipped}"
        )

    # Re-assert delta_logvar == 0 so the residual posterior still reduces to the prior in its
    # variance component. Deliberately NOT _zero_init_delta_heads(), which would wipe the freshly
    # loaded delta_mu_head; the raw residual decoder head is new (already zero) and untouched.
    model.zero_init_delta_logvar_head()
    return {"loaded": loaded, "missing": missing, "skipped": skipped}


# =============================================================================
# Trainer
# =============================================================================
class GraphModelVaeTebRawV4Trainer(GraphModelVaeTebLagAttnV3Trainer):
    """Experiment driver for :class:`SeqVaeRawV4`.

    Reuses v3's ``_select_ddp_strategy`` (plain ``'ddp'`` under learned variance +
    ``freeze_unused_attn_proj``; the always-used front ends add no unused-parameter risk). Overrides
    ``_build_model_kwargs`` to add the nested ``frontend`` block + raw geometry, ``create_model`` to
    build the raw model and wire the v3 warm-start, and ``train_model`` to register the raw plotting
    callback + a raw checkpoint filename.
    """

    def _build_model_kwargs(self) -> Dict[str, Any]:
        """Build v3's flat kwargs, then add the raw front end + geometry constructor args.

        The inherited (v3) sweep discovers constructor args from
        ``inspect.signature(SeqVaeLagAttnV3.__init__)`` and therefore silently drops ``frontend`` /
        ``raw_len`` / ``decimation``. Those flat v3 kwargs are still correct -- they reach
        :class:`SeqVaeRawV4` through ``**v3_kwargs`` and its forced ``sequence_length = geometry.t``
        -- so we only need to inject the raw-specific ones and re-sweep against the raw signature.
        """
        import inspect

        kwargs = super()._build_model_kwargs()
        vae_cfg = self.config.get("model_config", {}).get("VAE_model", {}) or {}

        # Nested raw front-end block + raw geometry (dropped by the v3 sweep).
        if vae_cfg.get("frontend") is not None:
            kwargs["frontend"] = dict(vae_cfg["frontend"])
        kwargs["raw_len"] = int(vae_cfg.get("raw_len", 5280))
        kwargs["decimation"] = int(vae_cfg.get("decimation", 16))

        # Forward any remaining flat key naming a real SeqVaeRawV4 arg (e.g. disable_source,
        # fhr_mean/std, up_mean/std). ``frontend`` is a nested group handled above; the raw
        # auxiliary-loss weights (lambda_lp/lambda_smooth/lowpass_scales) are trainer hparams, not
        # constructor args, and are (correctly) not in the raw signature so they are skipped here.
        valid_params = set(inspect.signature(SeqVaeRawV4.__init__).parameters)
        nested_groups = {"horizon_refine", "encoder", "frontend"}
        for name, value in vae_cfg.items():
            if name in kwargs or name in nested_groups or name == "init_weights":
                continue
            if name in valid_params and value is not None:
                kwargs[name] = value
        return kwargs

    def create_model(self) -> None:
        """Instantiate :class:`SeqVaeRawV4`, optionally warm-start from v3, and wrap it."""
        model_kwargs = self._build_model_kwargs()
        _fe = model_kwargs.get("frontend", {}) or {}
        logger.info(
            "Building SeqVaeRawV4 with kwargs: "
            + ", ".join(f"{k}={v}" for k, v in model_kwargs.items() if k != "frontend")
            + f", frontend=<{len(_fe)} keys>"
        )
        self.pytorch_model = SeqVaeRawV4(**model_kwargs)
        if not self.pytorch_model.causal_norm:
            logger.warning(
                "causal_norm=False: the encoders' GroupNorm pools statistics across time, so "
                "H_y[t] depends on Y[>t] and kld_raw is NOT a transfer-entropy surrogate."
            )

        model_cfg = self.config.get("model_config", {}) or {}
        vae_cfg = model_cfg.get("VAE_model", {}) or {}

        warm_start = model_cfg.get("warm_start_from")
        if warm_start:
            warm_start_from_v3(self.pytorch_model, str(warm_start))

        self.checkpoint = model_cfg.get("core_model_checkpoint")
        if self.checkpoint is not None:
            from model.vae_teb_prediction.model.vae_teb_lag_attn_trfr import (
                check_model_class,
            )
            from train.graph_models_utils import load_checkpoint_strict

            blob = torch.load(str(self.checkpoint), map_location="cpu", weights_only=False)
            check_model_class(blob, SeqVaeRawV4.__name__)
            if load_checkpoint_strict(model=self.pytorch_model, checkpoint=blob) is None:
                raise RuntimeError(
                    f"could not align core_model_checkpoint {self.checkpoint!r} into "
                    f"SeqVaeRawV4 (no matching module keys)."
                )
            logger.info(f"Model loaded from checkpoint: {self.checkpoint}")

        trainer_hparams = {
            "lr": self.lr,
            "lr_milestones": self.lr_milestones,
            "kld_beta": vae_cfg.get("kld_beta", 0.01),
            "lambda_full": vae_cfg.get("lambda_full", 1.0),
            "lambda_base": vae_cfg.get("lambda_base", 0.5),
            "beta_schedule": vae_cfg.get("beta_schedule"),
            # G7: learned observation variance is the raw model's only supported likelihood.
            "likelihood": vae_cfg.get("likelihood", "gaussian_nll"),
            "sigma_obs": vae_cfg.get("sigma_obs", "learned"),
            "free_bits": vae_cfg.get("free_bits", 0.0),
            "detach_baseline_in_full": vae_cfg.get("detach_baseline_in_full", True),
            "lambda_lag": vae_cfg.get("lag_smoothness_lambda", 0.0),
            # v4-specific raw auxiliary-loss weights (§10).
            "lambda_lp": vae_cfg.get("lambda_lp", 0.5),
            "lambda_smooth": vae_cfg.get("lambda_smooth", 0.1),
            "lowpass_scales": vae_cfg.get("lowpass_scales", [4, 16, 32, 60]),
            "loss_spike_skip": vae_cfg.get("loss_spike_skip", {}) or {},
        }
        self.pl_model = SeqVaeRawV4Pl(
            self.pytorch_model,
            lr=self.lr,
            lr_milestones=self.lr_milestones,
            model_kwargs=model_kwargs,
        )
        self.apply_config_hyperparameters(trainer_hparams, self.pl_model)

    def train_model(self, train_dataloader, validation_dataloader):
        """Build callbacks + Lightning Trainer and run ``trainer.fit`` (raw plotting + CSV)."""
        callbacks_cfg = self.config.get("advanced_config", {}).get("callbacks", {})
        self.metrics_callback = MetricsLoggingCallback(tracked_metrics=_TRACKED_METRICS_V4)
        self.metrics_csv_callback = MetricsHistoryCsvCallback(
            source=self.metrics_callback, output_dir=self.train_results_dir
        )
        self.loss_plot_callback = LossPlotCallback(
            output_dir=self.train_results_dir,
            plot_frequency=self.config["general_config"].get("plot_frequency", 1),
            mlflow_logger=self.mlflow_logger,
        )
        self.hyperparam_callback = HyperparameterLoggingCallback(
            output_dir=self.train_results_dir, plot_frequency=10
        )
        self.checkpoint_callback = ModelCheckpoint(
            dirpath=self.model_checkpoint_dir,
            monitor="val/total_loss",
            filename="raw-v4-epoch={epoch:02d}",
            save_top_k=callbacks_cfg.get("model_checkpoint", {}).get("save_top_k", 3),
            mode=callbacks_cfg.get("model_checkpoint", {}).get("mode", "min"),
        )
        callback_list: List[Callback] = [
            self.metrics_callback,
            self.metrics_csv_callback,
            self.loss_plot_callback,
            self.hyperparam_callback,
        ]

        plot_cfg = callbacks_cfg.get("raw_plotting", {}) or {}
        if plot_cfg.get("enabled", True):
            # Lazy import (matplotlib + utils.style): a bare ``import trainer_raw_v4`` for unit tests
            # / checkpoint tooling must not pay for it, and it sidesteps the utils-package shadowing.
            from model.vae_teb_prediction.model.model_raw.plotting_callback_raw_v4 import (
                RawV4PlotCallback,
            )

            self.raw_plot_callback = RawV4PlotCallback(
                output_dir=self.train_results_dir,
                plot_frequency=int(plot_cfg.get("plot_frequency", 5)),
                num_examples=int(plot_cfg.get("num_examples", 2)),
                file_format=str(plot_cfg.get("file_format", "pdf")),
                mlflow_logger=self.mlflow_logger,
                forecast_anchor_frac=float(plot_cfg.get("forecast_anchor_frac", 0.6)),
            )
            callback_list.append(self.raw_plot_callback)

        callback_list.append(self.checkpoint_callback)

        trainer_cfg = self.config.get("advanced_config", {}).get("trainer", {})
        logger_reference = self.lightning_loggers if self.lightning_loggers else True
        trainer_kwargs: Dict[str, Any] = {
            "max_epochs": self.epochs_num,
            "callbacks": callback_list,
            "default_root_dir": self.train_results_dir,
            "accumulate_grad_batches": self.accumulate_grad_batches,
            "precision": trainer_cfg.get("precision", "32-true"),
            "deterministic": trainer_cfg.get("deterministic", False),
            "benchmark": trainer_cfg.get("benchmark", True),
            "gradient_clip_val": trainer_cfg.get("gradient_clip_val"),
            "gradient_clip_algorithm": trainer_cfg.get("gradient_clip_algorithm", "norm"),
            "enable_checkpointing": True,
            "log_every_n_steps": 1,
            "num_sanity_val_steps": 0,
            "use_distributed_sampler": True,
            "sync_batchnorm": len(self.cuda_devices) > 1,
            "enable_progress_bar": True,
            "profiler": SimpleProfiler(dirpath=self.train_results_dir),
            "logger": logger_reference,
        }
        if torch.cuda.is_available():
            vae_cfg = self.config.get("model_config", {}).get("VAE_model", {}) or {}
            strategy = self._select_ddp_strategy(
                num_devices=len(self.cuda_devices),
                likelihood=str(vae_cfg.get("likelihood", "gaussian_nll")),
                sigma_obs=vae_cfg.get("sigma_obs", "learned"),
                curriculum_enabled=False,
                head_structured_latent=bool(vae_cfg.get("head_structured_latent", False)),
                freeze_unused_attn_proj=bool(self.pytorch_model.frozen_attn_proj),
            )
            logger.info(f"DDP strategy: {strategy}")
            trainer_kwargs.update(
                {"accelerator": "gpu", "devices": self.cuda_devices, "strategy": strategy}
            )
        else:
            trainer_kwargs.update({"accelerator": "cpu", "devices": 1})
        trainer = pl.Trainer(**trainer_kwargs)
        trainer.fit(self.pl_model, train_dataloader, validation_dataloader)
        return trainer


# =============================================================================
# Entry point
# =============================================================================
_DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "config_raw_v4.yaml"
)


def main(config_path: str = _DEFAULT_CONFIG) -> None:
    """Build raw data loaders, the raw model, the trainer, and run ``fit``."""
    np.random.seed(42)
    torch.manual_seed(42)
    start_time = time.time()

    with open(config_path) as f:
        config = yaml.safe_load(f)

    dataset_config = config.get("dataset_config")
    dataloader_config = dataset_config.get("dataloader_config")
    dataset_kwargs = dataloader_config.get("dataset_kwargs")
    normalize_fields = dataloader_config.get("normalize_fields")
    stat_path = dataset_config.get("stat_path")
    if stat_path is None:
        raise ValueError("stat_path must be provided")

    train_dataloader = create_optimized_dataloader(
        hdf5_files=dataset_config.get("vae_train_datasets", []),
        batch_size=config["general_config"]["batch_size"]["train"],
        num_workers=dataloader_config.get("num_workers", 4),
        shuffle=True,
        stats_path=stat_path,
        normalize_fields=normalize_fields,
        prefetch_factor=dataloader_config.get("prefetch_factor", 2),
        rank=0,
        world_size=1,
        **dataset_kwargs,
    )
    validation_dataloader = create_optimized_dataloader(
        hdf5_files=dataset_config.get("vae_test_datasets", []),
        batch_size=config["general_config"]["batch_size"]["test"],
        num_workers=dataloader_config.get("num_workers", 4),
        shuffle=False,
        stats_path=stat_path,
        normalize_fields=normalize_fields,
        prefetch_factor=dataloader_config.get("prefetch_factor", 2),
        rank=0,
        world_size=1,
        **dataset_kwargs,
    )

    graph_model = GraphModelVaeTebRawV4Trainer(config_file_path=config_path)
    graph_model.setup_config()
    graph_model.create_model()
    graph_model.train_model(train_dataloader, validation_dataloader)
    logger.info(f"Training completed in {(time.time() - start_time) / 60:.2f} minutes.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=_DEFAULT_CONFIG,
        help="Path to the YAML config (default: config_raw_v4.yaml).",
    )
    args = parser.parse_args()
    main(args.config)
