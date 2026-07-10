r"""Lightning wrapper + Graph-model trainer for :class:`SeqVaeLagAttnV3`.

Parallel to :mod:`model.vae_teb_prediction.model.trainer_lag_attn_v1`, which stays
byte-unchanged. This trainer adds the four things v3 needs on top of v1's loop:

* **Learned observation variance (G7).** ``sigma_obs='learned'`` makes ``compute_loss``
  consume the decoder log-variance heads, so the model emits a genuine per-point predictive
  distribution and :meth:`_select_ddp_strategy` resolves to plain ``'ddp'``.
* **Source-permutation control (G6).** Every ``perm_every_n_batches``-th step adds
  :math:`\lambda_{\mathrm{perm}} L_{\mathrm{perm}}` to the loss, where
  :math:`L_{\mathrm{perm}} = \mathrm{KL}(q(z \mid Y, \pi(U)) \,\|\, p(z \mid Y))` under a
  batch derangement :math:`\pi`.
* **Raw/train/active KL reporting (G4)** plus :math:`K_{\mathrm{shuffled}}`, surfaced to
  MLflow, the loss plots, and a metrics-history CSV.
* **Warm-start from a v1 checkpoint**, aligned submodule-by-submodule.

**Why automatic optimization, not the two-backward manual loop.**
Lightning runs ``training_step`` *inside* ``DistributedDataParallel.forward()`` (via forward
redirection), and ``DDPStrategy.pre_backward`` calls ``prepare_for_backward`` on **every**
``manual_backward``. Under plain ``'ddp'`` (``find_unused_parameters=False``) the reducer then
expects all parameters marked ready on each backward -- but an ``encode_only``-based
``L_perm`` backward never touches the decoders, so it raises. Manual optimization additionally
makes Lightning reject ``Trainer(gradient_clip_val>0)`` and ``accumulate_grad_batches != 1``
outright, forcing a hand-rolled clip, accumulation boundary, LR-scheduler step, and
circuit breaker.

Instead the control is *fused* into the single main forward:
:meth:`SeqVaeLagAttnV3.perm_kl_from_forward` permutes the already-computed ``source_state``
along the batch axis, which is exactly equivalent to re-encoding a permuted source stream
because the source path is batch-independent. One forward, one backward; a parameter used
twice in one graph shares a single ``AccumulateGrad`` node and is marked ready exactly once.

Usage:
    python -m model.vae_teb_prediction.model.trainer_lag_attn_v3 \\
        --config model/vae_teb_prediction/model/config_lag_attn_v3.yaml
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import lightning as pl
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.profilers import SimpleProfiler
from loguru import logger

from hdf5_dataset.hdf5_dataset import create_optimized_dataloader
from model.vae_teb_prediction.model.trainer_lag_attn_v1 import (
    GraphModelVaeTebLagAttnV1Trainer,
    SeqVaeLagAttnPl,
)
from model.vae_teb_prediction.model.vae_teb_lag_attn_v3 import SeqVaeLagAttnV3
from train.callbacks import (
    HyperparameterLoggingCallback,
    LossPlotCallback,
    MetricsLoggingCallback,
)
from train.graph_models_utils import _prepare_checkpoint_state_dict

#: Metric suffixes logged by :meth:`SeqVaeLagAttnV3Pl.compute_loss_and_metrics` that
#: ``MetricsLoggingCallback`` must be told about explicitly -- it hardcodes its list.
_V3_METRIC_SUFFIXES: Tuple[str, ...] = (
    "total_loss", "main_loss", "feat_loss", "base_loss",
    "kld_loss", "kld_raw", "kld_train", "kld_active_frac",
    "perm_loss", "kld_shuffled", "kld_shuffled_ratio",
    "feat_loss_shuffled", "shuffle_penalty",
    "mean_logvar_full", "mean_logvar_base",
    "pred_gap", "delta_mu_rms", "mu_post_prior_gap_rms",
)

#: The stage-prefixed names ``LightningModelBase._log_metrics`` actually emits, plus ``lr``
#: (v1's default tracked list asks for ``learning_rate``, which nothing ever logs).
_TRACKED_METRICS: Tuple[str, ...] = tuple(
    f"{stage}/{name}" for stage in ("train", "val") for name in _V3_METRIC_SUFFIXES
) + ("lr", "kld_beta")

#: Parameters a fresh v3 model owns that a v1 checkpoint cannot supply.
_EXPECTED_MISSING_PREFIXES: Tuple[str, ...] = (
    "posterior_head.delta_logvar_head.",  # G1 residual head (zero-init, stays zero)
    "lag_attn.lag_score_bias",            # G5, only when lag_bias_init='alibi_decay'
)
#: Parameters a v1 checkpoint carries that a residual-posterior v3 model has deleted.
_EXPECTED_UNEXPECTED_PREFIXES: Tuple[str, ...] = ("posterior_head.logvar_post_head.",)


# =============================================================================
# Callbacks
# =============================================================================
class MetricsHistoryCsvCallback(Callback):
    """Dump :class:`MetricsLoggingCallback`'s in-memory history to a CSV at fit end.

    ``MetricsLoggingCallback`` accumulates ``self.history`` but never writes it anywhere and
    nothing in the v1 trainer consumes ``as_dict()``. Rather than edit the shared
    ``train/callbacks.py``, v3 attaches this thin writer so the new KL/perm diagnostics land
    in a file that survives the run.
    """

    def __init__(self, source: MetricsLoggingCallback, output_dir: str) -> None:
        """Initialize.

        Args:
            source: The ``MetricsLoggingCallback`` whose history to serialise.
            output_dir: Directory receiving ``metrics_history.csv``.
        """
        super().__init__()
        self.source = source
        self.output_dir = output_dir

    def _write(self) -> Optional[str]:
        import pandas as pd

        history = self.source.as_dict()
        if not history:
            return None
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, "metrics_history.csv")
        frame = pd.DataFrame(history)
        frame.insert(0, "epoch", range(len(frame)))
        frame.to_csv(path, index=False)
        return path

    def on_validation_epoch_end(self, trainer, pl_module) -> None:  # type: ignore[override]
        """Rewrite the CSV each validation epoch so a killed run still leaves history."""
        if trainer.is_global_zero and not trainer.sanity_checking:
            self._write()

    def on_fit_end(self, trainer, pl_module) -> None:  # type: ignore[override]
        """Write the final history CSV on rank zero."""
        if trainer.is_global_zero:
            path = self._write()
            if path:
                logger.info(f"metrics history written to {path}")


# =============================================================================
# Lightning wrapper
# =============================================================================
class SeqVaeLagAttnV3Pl(SeqVaeLagAttnPl):
    r"""Lightning wrapper for :class:`SeqVaeLagAttnV3`.

    Inherits v1's ``torch.compile`` bypass, source-stream builder, :math:`\beta` schedule,
    residual diagnostics, and rank-synced loss-spike circuit breaker. Overrides the loss
    builder to add the v3 reporting keys and the fused source-permutation control, and the
    training step so the spike statistic is computed on the **main** loss only -- otherwise
    the periodic :math:`L_{\mathrm{perm}}` jump (and the NLL scale change from learned
    variance) would trip the breaker.
    """

    prog_bar_metrics: Tuple[str, ...] = ("total_loss", "feat_loss", "kld_raw")

    def __init__(
        self,
        base_model: nn.Module,
        *,
        lr: float = 1e-4,
        lr_milestones: Optional[Iterable[int]] = None,
        weight_decay: float = 1e-4,
        module_name: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the wrapper.

        Args:
            base_model: The :class:`SeqVaeLagAttnV3` instance to wrap.
            lr: Learning rate stored in ``self.hparams``.
            lr_milestones: Optional epoch milestones for the LR scheduler.
            weight_decay: AdamW weight decay applied across parameters.
            module_name: Friendly name used in logs.
            model_kwargs: The exact constructor kwargs used to build ``base_model``. Written
                into every checkpoint so the testing pipeline's version-agnostic load path can
                rebuild the architecture without a config file (S4-T06).
        """
        super().__init__(
            base_model,
            lr=lr,
            lr_milestones=lr_milestones,
            weight_decay=weight_decay,
            module_name=module_name,
        )
        self._model_kwargs: Dict[str, Any] = dict(model_kwargs or {})
        # CPU generator for the batch derangement. Seeded per rank so the shuffles differ
        # across ranks (their data differs); the *schedule* is rank-invariant, which is what
        # DDP actually requires.
        self._perm_generator = torch.Generator()
        self._perm_generator.manual_seed(1234 + int(getattr(self, "global_rank", 0) or 0))

    # ------------------------------------------------------------------
    # Source-permutation control (G6)
    # ------------------------------------------------------------------
    def _sync_perm_decision(self, do_perm: bool, device: torch.device) -> bool:
        """MIN-reduce ``do_perm`` so no rank runs the control alone.

        ``batch_idx`` is already rank-invariant, but a rank whose local batch is degenerate
        (``B < 2``) cannot be deranged. Without this reduction that rank would build a
        different autograd graph and the DDP all-reduce in ``backward`` would deadlock.
        """
        if not (dist.is_available() and dist.is_initialized()):
            return do_perm
        flag = torch.tensor([1.0 if do_perm else 0.0], device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        return bool(flag.item() > 0.0)

    def _should_run_perm(self, batch_idx: int, batch_size: int, stage: str) -> bool:
        r"""Decide whether to evaluate the source-permutation control on this step.

        The control runs as a **readout** regardless of ``lambda_perm``; the weight only
        decides whether it also enters the loss. Validation runs it on every step --
        ``val/kld_shuffled`` and ``val/feat_loss_shuffled`` are the headline G6 diagnostics,
        and under ``no_grad`` they are cheap. Training subsamples it on a rank-invariant
        ``batch_idx`` schedule.

        A batch too small to derange is skipped; the decision is MIN-reduced across ranks by
        :meth:`_sync_perm_decision` so no rank branches alone.
        """
        if batch_size < 2:
            return False
        if stage != "train":
            return True
        every = max(int(self.orig_model.perm_every_n_batches), 1)
        return batch_idx % every == 0

    # ------------------------------------------------------------------
    # Loss + metrics
    # ------------------------------------------------------------------
    def compute_loss_and_metrics(
        self, batch: Any, batch_idx: int, stage: str
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        r"""Forward pass, v3 loss, and the unified metrics dict.

        Adapted from :meth:`SeqVaeLagAttnPl.compute_loss_and_metrics` rather than delegating
        to it: v3 needs the ``forward_outputs`` dict for the fused permutation control, and
        v1's builder returns only ``(loss, metrics)``. Re-running the forward to recover it
        would double the DDP forward count -- exactly the failure this design avoids.

        Returns:
            ``(train_loss, metrics)`` where ``train_loss = main_loss + lambda_perm * L_perm``
            on scheduled steps and ``main_loss`` otherwise. ``metrics['main_loss']`` always
            carries the perm-free value, which is what the circuit breaker watches.
        """
        y_st = batch.fhr_st
        y_ph = batch.fhr_ph
        u_stream = self._build_source_stream(batch)
        weight = getattr(batch, "weight", None)

        forward_outputs = self.model(y_st, y_ph, u_stream)

        beta = self._resolve_beta(self.current_epoch)
        lambda_full = float(self.hparams.get("lambda_full", 1.0))
        lambda_base = float(self.hparams.get("lambda_base", 0.5))
        likelihood = str(self.hparams.get("likelihood", "gaussian_nll"))
        sigma_obs = self.hparams.get("sigma_obs", "learned")
        if not isinstance(sigma_obs, str):
            sigma_obs = float(sigma_obs)
        free_bits = float(self.hparams.get("free_bits", 0.0))
        detach_baseline_in_full = bool(self.hparams.get("detach_baseline_in_full", False))
        lambda_lag = float(self.hparams.get("lambda_lag", 0.0))

        loss_dict = self.orig_model.compute_loss(
            forward_outputs=forward_outputs,
            y_st=y_st,
            y_ph=y_ph,
            weight=weight,
            beta=beta,
            lambda_full=lambda_full,
            lambda_base=lambda_base,
            likelihood=likelihood,
            sigma_obs=sigma_obs,
            free_bits=free_bits,
            detach_baseline_in_full=detach_baseline_in_full,
            lambda_lag=lambda_lag,
        )
        main_loss = loss_dict["total_loss"]

        diag = self._compute_residual_diagnostics(
            forward_outputs=forward_outputs, weight=weight
        )
        kld_raw = loss_dict["kld_raw"]

        metrics: Dict[str, Any] = {
            "total_loss": main_loss,
            "main_loss": main_loss.detach(),
            "feat_loss": loss_dict["feat_loss"],
            "base_loss": loss_dict["base_loss"],
            "kld_loss": loss_dict["kld_loss"],
            # G4 -- only kld_raw may be read as a TE surrogate; kld_train is the
            # free-bit-floored quantity that actually enters total_loss.
            "kld_raw": kld_raw,
            "kld_train": loss_dict["kld_train"],
            "kld_active_frac": loss_dict["kld_active_frac"],
            "kld_beta": beta,
            "lambda_full": lambda_full,
            "lambda_base": lambda_base,
            # G7 -- watch these for variance collapse (sigma^2 -> exp(-5)).
            "mean_logvar_full": loss_dict["mean_logvar_full"],
            "mean_logvar_base": loss_dict["mean_logvar_base"],
            "mu_prior_sat_frac": forward_outputs["mu_prior_sat_frac"],
            "delta_mu_sat_frac": forward_outputs["delta_mu_sat_frac"],
            "delta_mu_rms": diag["delta_mu_rms"],
            "mu_post_prior_gap_rms": diag["mu_post_prior_gap_rms"],
            "pred_gap": loss_dict["base_loss"] - loss_dict["feat_loss"],
            "lag_smoothness": loss_dict["lag_smoothness"],
        }

        # --- G6: fused source-permutation control ---------------------------
        lambda_perm = float(self.orig_model.lambda_perm)
        do_perm = self._sync_perm_decision(
            self._should_run_perm(batch_idx, y_st.size(0), stage), y_st.device
        )
        train_loss = main_loss
        if do_perm:
            # Only build an autograd graph for the control when it actually enters the loss.
            optimise_perm = stage == "train" and lambda_perm > 0.0
            with torch.set_grad_enabled(optimise_perm and torch.is_grad_enabled()):
                perm = self.orig_model.perm_kl_from_forward(
                    forward_outputs, weight=weight, generator=self._perm_generator
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

            # Prediction-space control. The KL-space one does not discriminate: a deranged UP
            # is still a UP, and the posterior -- trained only on matched pairs -- reacts to it
            # out of distribution, so K_shuffled >= K_true even when the source is genuinely
            # used. The forecast tells the truth. A model that exploits the source has
            #     feat_loss  <  base_loss  <  feat_loss_shuffled,
            # i.e. a wrong source is worse than no source at all.
            with torch.no_grad():
                permuted = self.orig_model.perm_forward_outputs(
                    forward_outputs, perm_index=perm["perm_index"]
                )
                shuffled = self.orig_model.compute_loss(
                    forward_outputs=permuted,
                    y_st=y_st,
                    y_ph=y_ph,
                    weight=weight,
                    compute_kld_loss=False,
                    beta=0.0,
                    lambda_full=lambda_full,
                    lambda_base=lambda_base,
                    likelihood=likelihood,
                    sigma_obs=sigma_obs,
                    detach_baseline_in_full=detach_baseline_in_full,
                )
            metrics["feat_loss_shuffled"] = shuffled["feat_loss"]
            metrics["shuffle_penalty"] = shuffled["feat_loss"] - loss_dict["feat_loss"]
        else:
            zero = torch.zeros_like(kld_raw)
            for key in ("perm_loss", "kld_shuffled", "kld_shuffled_ratio",
                        "feat_loss_shuffled", "shuffle_penalty"):
                metrics[key] = zero

        return train_loss, metrics

    def training_step(self, batch: Any, batch_idx: int):  # type: ignore[override]
        """Forward + loss, gated by a spike check computed on the **main** loss.

        Identical in spirit to :meth:`SeqVaeLagAttnPl.training_step`, except the EMA and the
        threshold see ``metrics['main_loss']`` rather than the returned ``total_loss``. The
        permutation control fires only every ``perm_every_n_batches`` steps, so including it
        would make the breaker's own statistic periodic and it would eventually skip every
        perm step. On a spike we return a finite zero-valued no-op tensor built from a
        parameter (Lightning rejects ``None`` under DDP, and the all-reduce must still run).
        """
        import math

        train_loss, metrics = self.compute_loss_and_metrics(batch, batch_idx, stage="train")

        cfg = self._spike_cfg
        main_value = float(metrics["main_loss"].detach().item())
        is_nonfinite = not math.isfinite(main_value)

        ema_before = self._spike_ema_loss
        seen_before = self._spike_batches_seen
        self._spike_batches_seen += 1

        is_spike = False
        if cfg["enabled"]:
            if is_nonfinite:
                is_spike = True
            elif seen_before < int(cfg["warmup_batches"]):
                pass  # priming the EMA
            elif ema_before is not None and ema_before > 0.0:
                if main_value > float(cfg["multiplier"]) * ema_before:
                    is_spike = True

        is_spike = self._sync_skip_decision_across_ranks(is_spike, device=train_loss.device)

        if not is_spike:
            m = float(cfg["ema_momentum"])
            self._spike_ema_loss = (
                main_value if ema_before is None else m * main_value + (1.0 - m) * ema_before
            )
        else:
            self._spike_skips_total += 1

        ema_for_log = self._spike_ema_loss if self._spike_ema_loss is not None else main_value
        metrics["spike_ema_loss"] = self._as_tensor(ema_for_log)
        metrics["spike_skipped"] = self._as_tensor(1.0 if is_spike else 0.0)
        metrics["spike_skips_total"] = self._as_tensor(self._spike_skips_total)
        self._log_metrics(metrics, stage="train", on_step=True)

        if is_spike:
            if cfg["warn_on_skip"]:
                ema_str = "n/a" if ema_before is None else f"{ema_before:.4e}"
                logger.warning(
                    "[spike-skip] batch_idx={} main_loss={:.4e} ema={} nonfinite={} "
                    "total_skips={}",
                    batch_idx, main_value, ema_str, is_nonfinite, self._spike_skips_total,
                )
            # Must touch EVERY trainable parameter: the forward above already armed
            # DDP's reducer, which expects one gradient hook per parameter. A
            # single-parameter anchor leaves the rest unreduced and the next iteration
            # raises ``Expected to have finished reduction in the prior iteration``.
            # ``nan_to_num`` keeps the value finite when a parameter has itself gone NaN.
            return sum(
                torch.nan_to_num(p).sum()
                for p in self.parameters()
                if p.requires_grad
            ) * 0.0

        return train_loss

    # ------------------------------------------------------------------
    # Checkpoint contract (S4-T06)
    # ------------------------------------------------------------------
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:  # type: ignore[override]
        """Stamp ``model_class`` / ``model_kwargs`` onto every Lightning checkpoint.

        ``TestRunner.from_checkpoint`` prefers rebuilding from ``model_kwargs`` (the
        version-agnostic path) and ``check_model_class`` guards against loading a v1 or v2
        blob under the v3 alias. A stock Lightning ``.ckpt`` carries neither field.
        """
        checkpoint["model_class"] = type(self.orig_model).__name__
        checkpoint["model_kwargs"] = dict(self._model_kwargs)


# =============================================================================
# Warm-start
# =============================================================================
def warm_start_from_v1(
    model: SeqVaeLagAttnV3, checkpoint_path: str, *, strict_expectations: bool = True
) -> Dict[str, List[str]]:
    r"""Load a v1 checkpoint into a v3 model, submodule-by-submodule.

    :func:`train.graph_models_utils.load_checkpoint_strict` cannot do this: it loads a
    candidate module only on a *perfect* key/shape bijection, and a v3 model never achieves
    one against a v1 blob (it gains ``delta_logvar_head`` and, in residual mode, drops
    ``logvar_post_head``). Called against a v1 blob it would align nothing, log a warning, and
    return ``None`` -- i.e. silently train from scratch.

    Because v3 *subclasses* v1 and keeps every submodule name, a filtered
    ``load_state_dict(strict=False)`` after the shared prefix-stripping aligns everything else
    exactly. :class:`CausalGroupNorm` is deliberately parameter-compatible with
    :class:`torch.nn.GroupNorm`, so ``causal_norm=True`` does not perturb the alignment.

    Note:
        Zero init-KL after warm-start holds only when the source checkpoint is a *freshly
        constructed* v1, whose ``delta_mu_head`` is zero-initialised. Warm-starting from a
        genuinely trained v1 loads a nonzero ``delta_mu_head``, so :math:`\mu_q \neq \mu_p`
        and :math:`K > 0` at step 0. That is expected, not a bug.

    Args:
        model: The freshly constructed v3 model to populate, mutated in place.
        checkpoint_path: Path to a v1 checkpoint (Lightning ``.ckpt`` or a
            ``{model_class, model_state_dict, model_kwargs}`` blob).
        strict_expectations: If True, raise when the missing/unexpected key sets contain
            anything beyond the known v1-vs-v3 architectural delta.

    Returns:
        ``{"loaded": [...], "missing": [...], "unexpected": [...]}`` parameter names.

    Raises:
        ValueError: If the checkpoint declares a ``model_class`` other than
            ``SeqVaeLagAttnV1``, if its state dict cannot be extracted, or if
            ``strict_expectations`` is set and an unexplained key appears.
    """
    blob = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

    # NOT check_model_class: that guard asserts the blob matches the *active* class and would
    # raise on the very cross-version load this function exists to perform.
    declared = blob.get("model_class") if isinstance(blob, dict) else None
    if declared is not None and str(declared) != "SeqVaeLagAttnV1":
        raise ValueError(
            f"warm_start_from expects a SeqVaeLagAttnV1 checkpoint, got model_class="
            f"{declared!r}. For a same-class reload use model_config.core_model_checkpoint."
        )

    state = _prepare_checkpoint_state_dict(blob, map_location="cpu")
    if state is None:
        raise ValueError(f"could not extract a state dict from {checkpoint_path!r}")

    incompatible = model.load_state_dict(state, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    loaded = sorted(set(model.state_dict()) - set(missing))

    def _explained(key: str, prefixes: Tuple[str, ...]) -> bool:
        return any(key.startswith(p) for p in prefixes)

    surprising_missing = [k for k in missing if not _explained(k, _EXPECTED_MISSING_PREFIXES)]
    surprising_unexpected = [
        k for k in unexpected if not _explained(k, _EXPECTED_UNEXPECTED_PREFIXES)
    ]

    logger.info(
        "warm-start from {}: aligned {} tensors, left {} v3-only tensors at their "
        "initialisation, ignored {} v1-only tensors",
        checkpoint_path, len(loaded), len(missing), len(unexpected),
    )
    if missing:
        logger.info("  v3-only (not warm-started): {}", ", ".join(missing))
    if unexpected:
        logger.info("  v1-only (ignored): {}", ", ".join(unexpected))

    if not loaded:
        raise ValueError(
            f"warm-start aligned ZERO tensors from {checkpoint_path!r}; the checkpoint's "
            "keys do not match this architecture. Check that the v3 model_kwargs mirror "
            "the v1 run (d_model, d_z, horizon_refine, encoder.extra_dilations, ...)."
        )
    if strict_expectations and (surprising_missing or surprising_unexpected):
        raise ValueError(
            "warm-start produced key differences beyond the known v1-vs-v3 delta.\n"
            f"  unexplained missing:    {surprising_missing}\n"
            f"  unexplained unexpected: {surprising_unexpected}"
        )

    # Re-assert delta_logvar == 0 so the residual posterior still reduces to the prior in its
    # variance component. Deliberately NOT _zero_init_delta_heads(), which would also wipe the
    # freshly loaded delta_mu_head and residual_decoder.mean_head.
    model.zero_init_delta_logvar_head()
    return {"loaded": loaded, "missing": missing, "unexpected": unexpected}


# =============================================================================
# Trainer
# =============================================================================
class GraphModelVaeTebLagAttnV3Trainer(GraphModelVaeTebLagAttnV1Trainer):
    """Experiment driver for :class:`SeqVaeLagAttnV3`.

    Reuses v1's ``_build_model_kwargs`` -- its ``inspect.signature`` forward-compat block
    already forwards any flat ``VAE_model`` key naming a real constructor argument, so v3's
    ``causal_norm`` / ``logvar_bound`` / ``posterior_logvar`` / ``delta_logvar_scale`` /
    ``kld_support`` / ``lambda_perm`` / ``perm_every_n_batches`` reach the model unchanged --
    and v1's ``_select_ddp_strategy``, which returns plain ``'ddp'`` exactly when the learned
    log-variance heads are consumed.

    ``create_model`` and ``train_model`` are re-implemented (not extended) because v1 resolves
    its model class through a module-level alias and must stay byte-unchanged.
    """

    @staticmethod
    def _select_ddp_strategy(  # type: ignore[override]
        num_devices: int,
        likelihood: str,
        sigma_obs: Any,
        curriculum_enabled: bool = False,
        *,
        head_structured_latent: bool = False,
        freeze_unused_attn_proj: bool = False,
    ) -> str:
        r"""Select the Lightning DDP ``strategy`` string, accounting for v3's extra flags.

        Plain ``'ddp'`` implies ``find_unused_parameters=False``, under which the reducer
        expects **every** parameter to be marked ready in every backward. v1's rule tracks only
        the decoder log-variance heads, which are consumed exactly when
        ``likelihood='gaussian_nll'`` and ``sigma_obs='learned'``.

        v3 adds a second starved parameter. With ``head_structured_latent=True`` the posterior
        consumes ``A_heads`` and the attention's output projection ``lag_attn.W_o`` feeds only
        the diagnostic ``attended_source`` key, so it never receives a gradient. v1 never hit
        this because its fixed ``sigma_obs`` already forced ``find_unused_parameters``; v3's
        learned variance removes that cover. Either freeze ``W_o``
        (``freeze_unused_attn_proj=True``, numerically a no-op -- see
        :class:`SeqVaeLagAttnV3`) or accept the slower strategy.

        Args:
            num_devices: Number of CUDA devices for the run.
            likelihood: ``'mse'`` or ``'gaussian_nll'``.
            sigma_obs: Observation-noise scalar, or the string ``'learned'``.
            curriculum_enabled: Retained for signature compatibility with v1; unused by v3.
            head_structured_latent: Whether the posterior consumes per-head summaries.
            freeze_unused_attn_proj: Whether ``lag_attn.W_o`` was frozen at construction.

        Returns:
            The Lightning ``strategy`` string.
        """
        if num_devices <= 1:
            return "auto"
        if curriculum_enabled:
            return "ddp_find_unused_parameters_true"
        logvar_heads_consumed = (
            likelihood == "gaussian_nll"
            and isinstance(sigma_obs, str)
            and sigma_obs == "learned"
        )
        attn_proj_starved = bool(head_structured_latent) and not bool(freeze_unused_attn_proj)
        if logvar_heads_consumed and not attn_proj_starved:
            return "ddp"
        return "ddp_find_unused_parameters_true"

    def _build_model_kwargs(self) -> Dict[str, Any]:
        """Same as v1's, but discovering constructor args from :class:`SeqVaeLagAttnV3`."""
        import inspect

        kwargs = super()._build_model_kwargs()
        vae_cfg = self.config.get("model_config", {}).get("VAE_model", {}) or {}
        valid_params = set(inspect.signature(SeqVaeLagAttnV3.__init__).parameters)
        nested_groups = {"horizon_refine", "encoder"}
        for name, value in vae_cfg.items():
            if name in kwargs or name in nested_groups or name == "init_weights":
                continue
            if name in valid_params and value is not None:
                kwargs[name] = value
        return kwargs

    def create_model(self) -> None:
        """Instantiate :class:`SeqVaeLagAttnV3`, optionally warm-start, and wrap it."""
        model_kwargs = self._build_model_kwargs()
        logger.info(
            "Building SeqVaeLagAttnV3 with kwargs: "
            + ", ".join(f"{k}={v}" for k, v in model_kwargs.items())
        )
        self.pytorch_model = SeqVaeLagAttnV3(**model_kwargs)
        if not self.pytorch_model.causal_norm:
            logger.warning(
                "causal_norm=False: the encoders' GroupNorm pools statistics across time, so "
                "H_y[t] depends on Y[>t] and kld_raw is NOT a transfer-entropy surrogate. "
                "This setting exists only for Sprint-0 golden parity against v1."
            )

        model_cfg = self.config.get("model_config", {}) or {}
        vae_cfg = model_cfg.get("VAE_model", {}) or {}

        warm_start = model_cfg.get("warm_start_from")
        if warm_start:
            warm_start_from_v1(self.pytorch_model, str(warm_start))

        self.checkpoint = model_cfg.get("core_model_checkpoint")
        if self.checkpoint is not None:
            from model.vae_teb_prediction.model.vae_teb_lag_attn_trfr import (
                check_model_class,
            )
            from train.graph_models_utils import load_checkpoint_strict

            blob = torch.load(str(self.checkpoint), map_location="cpu", weights_only=False)
            check_model_class(blob, SeqVaeLagAttnV3.__name__)
            if load_checkpoint_strict(model=self.pytorch_model, checkpoint=blob) is None:
                raise RuntimeError(
                    f"could not align core_model_checkpoint {self.checkpoint!r} into "
                    f"SeqVaeLagAttnV3 (no matching module keys)."
                )
            logger.info(f"Model loaded from checkpoint: {self.checkpoint}")

        trainer_hparams = {
            "lr": self.lr,
            "lr_milestones": self.lr_milestones,
            "kld_beta": vae_cfg.get("kld_beta", 0.01),
            "lambda_full": vae_cfg.get("lambda_full", 1.0),
            "lambda_base": vae_cfg.get("lambda_base", 0.5),
            "beta_schedule": vae_cfg.get("beta_schedule"),
            # G7: learned observation variance is the v3 default.
            "likelihood": vae_cfg.get("likelihood", "gaussian_nll"),
            "sigma_obs": vae_cfg.get("sigma_obs", "learned"),
            "free_bits": vae_cfg.get("free_bits", 0.0),
            "detach_baseline_in_full": vae_cfg.get("detach_baseline_in_full", False),
            "lambda_lag": vae_cfg.get("lag_smoothness_lambda", 0.0),
            "loss_spike_skip": vae_cfg.get("loss_spike_skip", {}) or {},
        }
        self.pl_model = SeqVaeLagAttnV3Pl(
            self.pytorch_model,
            lr=self.lr,
            lr_milestones=self.lr_milestones,
            model_kwargs=model_kwargs,
        )
        self.apply_config_hyperparameters(trainer_hparams, self.pl_model)

    def train_model(self, train_dataloader, validation_dataloader):
        """Build callbacks + Lightning Trainer and run ``trainer.fit``."""
        callbacks_cfg = self.config.get("advanced_config", {}).get("callbacks", {})
        self.metrics_callback = MetricsLoggingCallback(tracked_metrics=_TRACKED_METRICS)
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
            filename="lag-attn-v3-epoch={epoch:02d}",
            save_top_k=callbacks_cfg.get("model_checkpoint", {}).get("save_top_k", 3),
            mode=callbacks_cfg.get("model_checkpoint", {}).get("mode", "min"),
        )
        callback_list: List[Callback] = [
            self.metrics_callback,
            self.metrics_csv_callback,
            self.loss_plot_callback,
            self.hyperparam_callback,
        ]

        plot_cfg = callbacks_cfg.get("lag_attn_plotting", {}) or {}
        if plot_cfg.get("enabled", True):
            # Imported lazily: the callback pulls in matplotlib and ``utils.style``, which
            # nothing else in the training path needs, and which a bare
            # ``import trainer_lag_attn_v3`` (unit tests, checkpoint tooling) should not pay
            # for. Also sidesteps the ``model/vae_teb_prediction/utils`` package shadowing the
            # repo-root ``utils`` whenever that directory lands early on ``sys.path``.
            from model.vae_teb_prediction.model.plotting_callback_lag_attn_v3 import (
                LagAttnV3PlotCallback,
            )

            self.lag_attn_plot_callback = LagAttnV3PlotCallback(
                output_dir=self.train_results_dir,
                plot_frequency=int(plot_cfg.get("plot_frequency", 5)),
                num_examples=int(plot_cfg.get("num_examples", 2)),
                file_format=str(plot_cfg.get("file_format", "pdf")),
                mlflow_logger=self.mlflow_logger,
                forecast_channels=tuple(
                    int(c) for c in plot_cfg.get("forecast_channels", [0, 43, 80])
                ),
                forecast_anchor_frac=float(plot_cfg.get("forecast_anchor_frac", 0.6)),
            )
            callback_list.append(self.lag_attn_plot_callback)

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
            # Automatic optimization keeps Lightning's clip + accumulation active; the
            # permutation control rides inside the single main backward.
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
    os.path.dirname(os.path.abspath(__file__)), "config_lag_attn_v3.yaml"
)


def main(config_path: str = _DEFAULT_CONFIG) -> None:
    """Build data loaders, model, trainer and run ``fit``."""
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

    graph_model = GraphModelVaeTebLagAttnV3Trainer(config_file_path=config_path)
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
        help="Path to the YAML config (default: config_lag_attn_v3.yaml).",
    )
    args = parser.parse_args()
    main(args.config)
