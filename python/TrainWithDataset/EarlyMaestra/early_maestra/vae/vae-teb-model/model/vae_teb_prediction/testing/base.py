"""
Base module for the VAE-TEB Lag-Attentive v1 testing pipeline.

This module provides the ``TestRunner`` class, which handles all common
boilerplate for testing the :class:`SeqVaeLagAttnV1` model: device
management, model construction from YAML config, checkpoint loading,
batch iteration, and output directory management.

The runner is intentionally **tied to the lag-attn v1 model I/O contract**
because every downstream collector/analysis expects the specific forward
dict produced by that model (``mu_full``, ``mu_base``, ``delta_mu_src``,
``attn_weights``, ``te_lag_map``, ``kld_per_t``, etc.). See
``new_architecture.md`` and ``vae_teb_lag_attn_v1.py`` for the authoritative
spec of inputs/outputs.

Example:
    >>> runner = TestRunner.from_checkpoint(
    ...     checkpoint_path="checkpoints/best.ckpt",
    ...     output_dir="results/",
    ...     config_path="model/vae_teb_prediction/model/config_lag_attn_v1.yaml",
    ... )
    >>> with runner.inference_mode():
    ...     for batch in runner.iter_batches(loader, max_samples=64):
    ...         outputs = runner.forward(batch)
    ...         y_plus = runner.build_future_target(batch)
"""

from __future__ import annotations

import inspect
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple, Union

import torch
import yaml

from train.graph_models_utils import load_checkpoint_strict

# Canonical model-class alias -- comment-toggle to switch v1 <-> v2 in one line
# (S6-T03). The active line is v1 (the committed default); uncomment the v2 line
# and comment the v1 line to evaluate a v2 checkpoint. The ``_old`` line is the
# legacy pre-refactor module kept only to align old (pre-375b50d) checkpoints;
# toggle it in (and comment the others) for that one case.
from model.vae_teb_prediction.model.vae_teb_lag_attn_v1 import SeqVaeLagAttnV1 as SeqVaeLagAttn  # ACTIVE (v1)
# from model.vae_teb_prediction.model.vae_teb_lag_attn_v2 import SeqVaeLagAttnV2 as SeqVaeLagAttn
# from model.vae_teb_prediction.model.vae_teb_lag_attn_old import SeqVaeLagAttnV1 as SeqVaeLagAttn
# The checkpoint model-class guard lives with v2; importing it is version-agnostic.
from model.vae_teb_prediction.model.vae_teb_lag_attn_v2 import check_model_class


def _lag_attn_kwargs_from_config(
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    r"""Build :class:`SeqVaeLagAttnV1` constructor kwargs from a YAML config dict.

    Reads ``cfg["model_config"]["VAE_model"]`` and forwards **every** field that
    names a real :class:`SeqVaeLagAttnV1` constructor argument, falling back to
    the constructor's own default for any argument the config omits. The valid
    argument set is discovered from the constructor signature via
    :func:`inspect.signature`, so architecture-defining flags -- e.g.
    ``head_structured_latent``, ``horizon_film``, ``horizon_depth``,
    ``horizon_kernel``, ``encoder_extra_dilations``, ``lag_bias_init`` -- flow
    through automatically and the rebuilt module's ``state_dict`` matches the
    checkpoint that recorded them. Fields that are not constructor arguments are
    ignored.

    This is what makes a checkpoint trained with a non-default architecture
    loadable: the synthetic trainer persists the *exact* resolved constructor
    kwargs in the checkpoint (``model_kwargs``) and
    ``run_pipeline_tests._synth_to_testing_config`` copies them verbatim under
    ``model_config.VAE_model``; the standalone testing YAML
    (``config_lag_attn_v1.yaml``) carries the same keys. Cherry-picking only a
    legacy subset (the previous behaviour) silently rebuilt the *default*
    architecture and made every non-default-arch checkpoint fail
    ``load_checkpoint_strict`` alignment.

    ``attention_grad_checkpoint`` is **forced to ``False``** regardless of the
    config value: checkpointing only helps the backward pass, and the test
    runner always runs under ``torch.inference_mode()``.

    Args:
        cfg: Parsed YAML config (e.g. from ``yaml.safe_load``).

    Returns:
        Keyword argument dictionary suitable for
        ``SeqVaeLagAttnV1(**kwargs)``.
    """
    vae_cfg: Dict[str, Any] = (
        (cfg.get("model_config", {}) or {}).get("VAE_model", {}) or {}
    )

    # Discover the real constructor arguments so any architecture flag the
    # checkpoint was trained with is honoured (and unknown config keys dropped).
    # ``init_weights`` is intentionally left at its constructor default (the
    # loaded checkpoint overwrites the weights regardless).
    params = inspect.signature(SeqVaeLagAttn.__init__).parameters
    kwargs: Dict[str, Any] = {}
    for name, param in params.items():
        if name in ("self", "init_weights"):
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if name in vae_cfg and vae_cfg[name] is not None:
            kwargs[name] = vae_cfg[name]
        elif param.default is not inspect.Parameter.empty:
            kwargs[name] = param.default

    # YAML lists -> tuples where the constructor stores tuples (mirrors the
    # coercion in ``train_minimal.build_model`` / ``evaluate_te._load_model``).
    clamp = kwargs.get("logvar_clamp")
    if isinstance(clamp, (list, tuple)) and len(clamp) == 2:
        kwargs["logvar_clamp"] = (float(clamp[0]), float(clamp[1]))
    extra_dil = kwargs.get("encoder_extra_dilations")
    if extra_dil is not None:
        kwargs["encoder_extra_dilations"] = tuple(int(x) for x in extra_dil)

    # Inference only: gradient checkpointing helps the backward pass alone.
    kwargs["attention_grad_checkpoint"] = False

    return kwargs


@dataclass
class TestRunner:
    """Minimal test harness for :class:`SeqVaeLagAttnV1`.

    Handles device management, model setup, batch iteration, and output
    directory organisation. Designed to be composed with analysis functions
    rather than inherited from.

    Attributes:
        model: The loaded :class:`SeqVaeLagAttnV1` instance (on ``device``).
        device: The torch device for inference.
        output_dir: Base directory for saving test results.
        warmup_steps: Number of initial timesteps to mask (mirrors
            ``model.warmup_period``).
        horizon: Forecast horizon ``H_d`` (mirrors ``model.horizon``).
        max_lag: Maximum causal lag ``L - 1`` (mirrors ``model.max_lag``).
        use_up_st: Whether the source stream concatenates ``up_st`` with
            ``up_ph`` (True) or uses ``up_ph`` only (False).

    Example:
        >>> runner = TestRunner.from_checkpoint(
        ...     "ckpt.ckpt", "results/", config_path="config_lag_attn_v1.yaml"
        ... )
        >>> with runner.inference_mode():
        ...     for batch in runner.iter_batches(loader):
        ...         outputs = runner.forward(batch)
    """

    model: SeqVaeLagAttn
    device: torch.device
    output_dir: Path
    warmup_steps: int = 30
    horizon: int = 30
    max_lag: int = 90
    use_up_st: bool = True

    # Private field to track if model is in eval mode.
    _in_inference: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        """Ensure ``output_dir`` is a ``Path`` object and exists."""
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        output_dir: Union[str, Path],
        config_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
    ) -> "TestRunner":
        """Create a runner by building the active-alias model and loading weights.

        Two build paths, tried in order (mirroring ``run_pipeline_v2``):

        1. **Version-agnostic** -- when the checkpoint carries its own
           ``model_kwargs`` (every ``save_checkpoint_v2`` /
           ``train_minimal.save_checkpoint`` file does), the model is rebuilt
           directly from those kwargs via the ``SeqVaeLagAttn`` alias. A
           :func:`check_model_class` guard runs first, so loading a v2 checkpoint
           under the v1 alias (or vice versa) fails with an actionable message
           **before** construction rather than a cryptic ``state_dict`` error.
           ``config_path`` is not required on this path.
        2. **Legacy config** -- for old (pre-``model_kwargs`` / ``_old``)
           checkpoints, ``config_path`` is parsed into constructor kwargs via
           :func:`_lag_attn_kwargs_from_config` (the original behaviour).

        Args:
            checkpoint_path: Path to a Lightning/PyTorch checkpoint file
                (``.ckpt`` or ``.pt``), possibly wrapped under ``_orig_model.`` /
                ``pytorch_model.``.
            output_dir: Directory for saving test results.
            config_path: Trainer YAML config. Optional when the checkpoint has
                ``model_kwargs``; required for the legacy path.
            device: Torch device to use. Auto-detects if None
                (``cuda:0`` or ``cpu``).

        Returns:
            Configured :class:`TestRunner` with the loaded model.

        Raises:
            FileNotFoundError: If the legacy path is taken and ``config_path`` is
                missing.
            ValueError: If the checkpoint's ``model_class`` does not match the
                active alias, or no build path is available.
            RuntimeError: If ``load_checkpoint_strict`` cannot align any
                candidate submodule with the checkpoint state dict.
        """
        from loguru import logger

        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Deserialise once so both the guard and the strict load see the same blob.
        blob = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        ckpt_kwargs = blob.get("model_kwargs") if isinstance(blob, dict) else None

        if ckpt_kwargs:
            # Path 1: rebuild from the checkpoint's own kwargs (version-agnostic).
            if config_path is not None:
                logger.warning(
                    "TestRunner.from_checkpoint: checkpoint carries its own "
                    "model_kwargs, so the supplied config_path={} is IGNORED "
                    "for architecture reconstruction (the checkpoint's kwargs "
                    "take precedence). Pass config_path=None to silence this, "
                    "or rebuild without model_kwargs to force the legacy "
                    "config-driven path.",
                    config_path,
                )
            check_model_class(blob, SeqVaeLagAttn.__name__)
            model_kwargs = dict(ckpt_kwargs)
            if isinstance(model_kwargs.get("logvar_clamp"), list):
                lv = model_kwargs["logvar_clamp"]
                model_kwargs["logvar_clamp"] = (float(lv[0]), float(lv[1]))
            source = "checkpoint model_kwargs"
        else:
            # Path 2: legacy checkpoints carry no kwargs; parse the YAML config.
            if config_path is None:
                raise ValueError(
                    "checkpoint has no 'model_kwargs'; a config_path is required "
                    "to rebuild the model (legacy path)."
                )
            cfg_path = Path(config_path)
            if not cfg_path.exists():
                raise FileNotFoundError(f"Config file not found: {cfg_path}")
            with cfg_path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            model_kwargs = _lag_attn_kwargs_from_config(cfg)
            source = f"config {cfg_path.name}"

        logger.info(
            "Building {} from {} with kwargs: "
            "d_model={}, d_z={}, c_y={}, c_u={}, use_up_st={}, horizon={}, "
            "max_lag={}, warmup_period={}",
            SeqVaeLagAttn.__name__, source,
            model_kwargs.get("d_model"), model_kwargs.get("d_z"),
            model_kwargs.get("c_y"), model_kwargs.get("c_u"),
            model_kwargs.get("use_up_st"), model_kwargs.get("horizon"),
            model_kwargs.get("max_lag"), model_kwargs.get("warmup_period"),
        )

        model = SeqVaeLagAttn(**model_kwargs)
        loaded = load_checkpoint_strict(model, blob)
        if loaded is None:
            raise RuntimeError(
                f"Failed to load {SeqVaeLagAttn.__name__} checkpoint "
                f"'{checkpoint_path}'. load_checkpoint_strict returned None; "
                f"inspect its log output for candidate-module alignment details "
                f"(common wrapper prefixes: '_orig_model.', 'model.', "
                f"'pytorch_model.')."
            )

        model.eval()
        model = model.to(device)

        return cls(
            model=model,
            device=device,
            output_dir=Path(output_dir),
            warmup_steps=int(getattr(model, "warmup_period", 30)),
            horizon=int(getattr(model, "horizon", 30)),
            max_lag=int(getattr(model, "max_lag", 90)),
            use_up_st=bool(getattr(model, "use_up_st", True)),
        )

    @classmethod
    def from_trainer(
        cls,
        trainer: Any,
        output_subdir: str = "test_results",
    ) -> "TestRunner":
        """Create a runner from an existing ``GraphModelVaeTebLagAttnV1Trainer``.

        Useful when you already have a trainer with a loaded model and want
        to run tests without reloading the checkpoint.

        Args:
            trainer: A ``GraphModelVaeTebLagAttnV1Trainer`` instance whose
                ``pytorch_model`` is a :class:`SeqVaeLagAttnV1`.
            output_subdir: Subdirectory name under the trainer's
                ``test_results_dir`` for results.

        Returns:
            Configured :class:`TestRunner` using the trainer's model.

        Raises:
            ValueError: If ``trainer.pytorch_model`` is None.
        """
        if trainer.pytorch_model is None:
            raise ValueError(
                "Trainer's pytorch_model is None. Call create_model() first."
            )

        cuda_devices = getattr(trainer, "cuda_devices", [])
        if cuda_devices and torch.cuda.is_available():
            device = torch.device(f"cuda:{cuda_devices[0]}")
        else:
            device = torch.device("cpu")

        output_dir = Path(
            getattr(trainer, "test_results_dir", "test_results")
        ) / output_subdir

        model = trainer.pytorch_model.to(device)

        return cls(
            model=model,
            device=device,
            output_dir=output_dir,
            warmup_steps=int(getattr(model, "warmup_period", 30)),
            horizon=int(getattr(model, "horizon", 30)),
            max_lag=int(getattr(model, "max_lag", 90)),
            use_up_st=bool(getattr(model, "use_up_st", True)),
        )

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @contextmanager
    def inference_mode(self):
        """Context manager for model evaluation with inference optimisations.

        Sets the model to ``.eval()`` and enables ``torch.inference_mode()``
        so forward passes run without gradient tracking and with all
        eval-mode layer behaviours (dropout off, BN running stats).

        Yields:
            None
        """
        was_training = self.model.training
        self.model.eval()
        self._in_inference = True

        try:
            with torch.inference_mode():
                yield
        finally:
            self._in_inference = False
            if was_training:
                self.model.train()

    def iter_batches(
        self,
        loader: Any,
        max_samples: Optional[int] = None,
    ) -> Iterator[Any]:
        """Iterate over batches with tensors moved to ``self.device``.

        Moves only the fields that the lag-attn v1 model needs:

        - ``fhr_st``, ``fhr_ph`` — FHR feature inputs
        - ``up_st``, ``up_ph`` — UP source stream components
        - ``up``, ``fhr`` — raw signals kept for plotting context

        Missing fields are silently skipped so the iterator tolerates
        datasets produced by ablated configs (e.g. ``use_up_st=False`` with
        ``up_st`` absent). Metadata fields (``guid``, ``epoch``, ``target``,
        ``cs_label``, ``bg_label``) are intentionally left on CPU because
        the collectors access them as Python scalars.

        Args:
            loader: PyTorch DataLoader yielding batch objects (normally
                an ``AttributeDict`` from ``CombinedHDF5Dataset``).
            max_samples: Maximum number of samples (not batches) to yield.
                ``None`` yields all.

        Yields:
            The batch object with tensors on ``self.device``.
        """
        processed = 0
        move_fields = ("fhr_st", "fhr_ph", "up_st", "up_ph", "up", "fhr")

        for batch in loader:
            if max_samples is not None and processed >= max_samples:
                break

            for fname in move_fields:
                t = getattr(batch, fname, None)
                if isinstance(t, torch.Tensor):
                    setattr(batch, fname, t.to(self.device, non_blocking=True))

            yield batch

            batch_size = int(batch.fhr_st.size(0))
            processed += batch_size

    def ensure_dir(self, subdir: str) -> Path:
        """Create and return an output subdirectory.

        Args:
            subdir: Name of the subdirectory under ``self.output_dir``.

        Returns:
            The created (or existing) subdirectory path.
        """
        path = self.output_dir / subdir
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ------------------------------------------------------------------
    # Forward / target construction
    # ------------------------------------------------------------------

    def _build_u_stream(self, batch: Any) -> torch.Tensor:
        """Assemble the 101- or 58-channel source stream for the lag-attn v1 model.

        Mirrors ``SeqVaeLagAttnPl._build_source_stream`` so the runner sees
        exactly the same source representation the trainer uses.

        Args:
            batch: Batch object with ``up_ph`` (and ``up_st`` when
                ``self.use_up_st`` is True).

        Returns:
            Source stream tensor of shape ``(B, T, 101)`` or ``(B, T, 58)``.

        Raises:
            RuntimeError: If required fields are missing from the batch.
        """
        up_ph = getattr(batch, "up_ph", None)
        if up_ph is None:
            raise RuntimeError(
                "Batch has no 'up_ph' field. Ensure 'up_ph' is in "
                "dataset_kwargs.load_fields of the config."
            )
        if not self.use_up_st:
            return up_ph

        up_st = getattr(batch, "up_st", None)
        if up_st is None:
            raise RuntimeError(
                "Model has use_up_st=True but batch has no 'up_st'. "
                "Add 'up_st' to load_fields, or rebuild the model with "
                "use_up_st=False and c_u=58."
            )
        return torch.cat([up_st, up_ph], dim=-1)

    def build_future_target(self, batch: Any) -> torch.Tensor:
        """Build the ground-truth future FHR feature trajectory ``Y_plus``.

        This is the authoritative target for every feature-forecast metric
        in the lag-attn v1 pipeline. It uses the same unfold formula that
        ``SeqVaeLagAttnV1.compute_loss`` applies internally.

        Given FHR features ``Y = cat(y_st, y_ph)`` of shape ``(B, T, 87)``,
        the future target at anchor ``t`` is ``Y[:, t+1 : t+1+H_d, :]``.
        Only anchors ``t in [0, T - H_d)`` have a full ``H_d``-step future;
        the returned tensor has shape ``(B, T - H_d, H_d, 87)``.

        Warmup masking is **not** applied here — collectors/metrics handle
        the ``[warmup, T - H_d)`` slicing themselves.

        Args:
            batch: Batch object with ``fhr_st`` and ``fhr_ph`` attributes
                already on ``self.device``.

        Returns:
            Future feature target of shape ``(B, T - H_d, H_d, 87)``.
        """
        Y = torch.cat([batch.fhr_st, batch.fhr_ph], dim=-1)  # (B, T, 87)
        Y_shift = Y[:, 1:, :]                                # (B, T-1, 87)
        H_d = int(self.horizon)
        # unfold: (B, T-H_d, 87, H_d) → permute to (B, T-H_d, H_d, 87)
        Y_plus = Y_shift.unfold(dimension=1, size=H_d, step=1)
        Y_plus = Y_plus.permute(0, 1, 3, 2).contiguous()
        return Y_plus

    def valid_anchor_range(self, seq_len: Optional[int] = None) -> Tuple[int, int]:
        """Return the valid anchor range ``[warmup, T - H_d)`` for forecast metrics.

        Args:
            seq_len: Sequence length ``T`` (defaults to the model's
                configured ``sequence_length``).

        Returns:
            ``(warmup, T_valid)`` where ``T_valid = T - H_d`` and ``warmup``
            is clamped into ``[0, T_valid]``.
        """
        if seq_len is None:
            seq_len = int(getattr(self.model, "sequence_length", 300))
        T_valid = max(seq_len - int(self.horizon), 0)
        warmup = min(int(self.warmup_steps), T_valid)
        return warmup, T_valid

    def forward(
        self,
        batch: Any,
        compute_loss: bool = False,
        beta: float = 1.0,
        lambda_full: float = 1.0,
        lambda_base: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """Run a forward pass on a batch and optionally compute the v1 loss.

        Args:
            batch: Batch object with ``fhr_st``, ``fhr_ph``, and the
                appropriate UP fields on ``self.device``.
            compute_loss: If True, also compute and attach
                ``outputs["loss_dict"]`` from ``model.compute_loss``.
            beta: KL weight for loss computation.
            lambda_full: Weight on the full-feature forecast loss.
            lambda_base: Weight on the baseline-only forecast loss.

        Returns:
            The full 19-key forward-output dict from
            :meth:`SeqVaeLagAttnV1.forward`, with an optional ``loss_dict``
            key when ``compute_loss=True``.
        """
        u_stream = self._build_u_stream(batch)
        outputs = self.model(
            y_st=batch.fhr_st,
            y_ph=batch.fhr_ph,
            u_stream=u_stream,
        )

        if compute_loss:
            outputs["loss_dict"] = self.model.compute_loss(
                forward_outputs=outputs,
                y_st=batch.fhr_st,
                y_ph=batch.fhr_ph,
                beta=beta,
                lambda_full=lambda_full,
                lambda_base=lambda_base,
            )

        return outputs
