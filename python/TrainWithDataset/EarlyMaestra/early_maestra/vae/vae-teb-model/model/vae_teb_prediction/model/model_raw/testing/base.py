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

# Canonical model-class alias for the raw port (S6-T02): the runner builds ``SeqVaeRawV4``, whose
# forward/target/mask contract is the raw fork of v3. The version-agnostic ``from_checkpoint`` path
# rebuilds from the checkpoint's stamped ``model_kwargs`` (which carry ``frontend``/``raw_len``/
# ``decimation``), so a raw checkpoint reconstructs its front ends + raw decoders exactly.
from model.vae_teb_prediction.model.model_raw.vae_teb_raw_v4 import SeqVaeRawV4 as SeqVaeLagAttn  # ACTIVE (raw v4)
# The checkpoint model-class guard lives with v2; importing it is version-agnostic.
from model.vae_teb_prediction.model.vae_teb_lag_attn_trfr import check_model_class

# Raw geometry / target / mask helpers -- the raw analogues of the feature ``build_future_target``
# and the source-stream assembly (§5.2/§5.5 of the roadmap).
from model.vae_teb_prediction.model.model_raw.geometry import GEOMETRY, derive_geometry
from model.vae_teb_prediction.model.model_raw.raw_targets import (
    build_future_target as build_raw_future_target,
)
from model.vae_teb_prediction.model.model_raw.raw_masks import frontend_mask


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

    #: Tell pytest this ``Test*``-named class is not a test case (it has an ``__init__``).
    __test__ = False

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

        Moves only the fields the raw ``SeqVaeRawV4`` model needs:

        - ``fhr``, ``up`` — raw $4$ Hz signals (the model inputs)
        - ``weight`` — the decimated validity signal (nearest-upsampled to the raw mask)

        The legacy feature fields (``fhr_st``/``fhr_ph``/``up_st``/``up_ph``) are also moved when
        present so the (kept, still feature-shaped in name only) plotting overlays that read raw
        ``fhr`` keep working; missing fields are silently skipped. Metadata fields (``guid``,
        ``epoch``, ``target``, ``cs_label``, ``bg_label``) are intentionally left on CPU because the
        collectors access them as Python scalars.

        Args:
            loader: PyTorch DataLoader yielding batch objects (normally
                an ``AttributeDict`` from ``CombinedHDF5Dataset``).
            max_samples: Maximum number of samples (not batches) to yield.
                ``None`` yields all.

        Yields:
            The batch object with tensors on ``self.device``.
        """
        processed = 0
        move_fields = ("fhr", "up", "weight", "fhr_st", "fhr_ph", "up_st", "up_ph")

        for batch in loader:
            if max_samples is not None and processed >= max_samples:
                break

            for fname in move_fields:
                t = getattr(batch, fname, None)
                if isinstance(t, torch.Tensor):
                    setattr(batch, fname, t.to(self.device, non_blocking=True))

            yield batch

            batch_size = int(batch.fhr.size(0))
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

    def geometry(self):
        """The raw/low-rate :class:`RawGeometry` this runner's model was built with.

        Prefers the model's own ``geometry`` attribute (stamped in
        ``SeqVaeRawV4.__init__``); falls back to deriving it from ``raw_len``/``decimation`` so a
        runner built from a bare ``nn.Module`` still resolves. This is what makes the runner work
        on both the tiny fixture geometry ($L_{\\mathrm{raw}}=512$) and production ($5280$).
        """
        geo = getattr(self.model, "geometry", None)
        if geo is not None:
            return geo
        return derive_geometry(
            int(getattr(self.model, "raw_len", GEOMETRY.raw_len)),
            int(getattr(self.model, "decimation", GEOMETRY.decimation)),
        )

    def _build_raw_input(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assemble the raw model inputs ``(fhr_raw, up_raw, mask)`` from a batch.

        Delegates to the model's own :meth:`SeqVaeRawV4._default_batch_to_inputs` so the runner
        sees exactly the ``(batch.fhr, batch.up, frontend_mask(batch.weight))`` triple the trainer
        feeds -- keeping the raw validity mask convention (nearest-upsampled decimated ``weight``)
        identical between training and evaluation.

        Args:
            batch: Batch object with ``fhr``, ``up``, ``weight`` on ``self.device``.

        Returns:
            ``(fhr_raw, up_raw, mask)`` each of shape $(B, L_{\\mathrm{raw}})$.

        Raises:
            RuntimeError: If the batch is missing a required raw field.
        """
        for fname in ("fhr", "up", "weight"):
            if getattr(batch, fname, None) is None:
                raise RuntimeError(
                    f"Batch has no '{fname}' field. Ensure it is in "
                    "dataset_kwargs.load_fields of the raw config."
                )
        return self.model._default_batch_to_inputs(batch)

    def build_future_target(self, batch: Any) -> torch.Tensor:
        r"""Build the ground-truth future raw-FHR waveform target $X^+$.

        This is the authoritative target for every raw-forecast metric. For each trained low-rate
        anchor $t \in [0, T_{\mathrm{valid}})$ it gathers the $2$-minute future raw block
        $X^+_{t,\tau,r} = x^y[\mathrm{future\_block\_start}(t) + D\tau + r]$ (crop-aligned; see
        ``raw_targets.build_future_target``), returning shape $(B, T_{\mathrm{valid}}, H, R)$.

        Warm-up masking is **not** applied here -- collectors/metrics handle the
        $[w, T-H)$ slicing themselves (mirroring the feature pipeline). Note the model's
        ``forward`` emits decoder tensors at the full $T$ anchor axis, so metric code must slice
        the prediction to $T_{\mathrm{valid}}$ to align with this target.

        Args:
            batch: Batch object with ``fhr`` on ``self.device``.

        Returns:
            Future raw-FHR target of shape $(B, T_{\mathrm{valid}}, H, R)$.
        """
        return build_raw_future_target(batch.fhr, self.geometry())

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
        """Run a raw forward pass on a batch and optionally compute the raw loss.

        Args:
            batch: Batch object with ``fhr``, ``up``, ``weight`` on ``self.device``.
            compute_loss: If True, also compute and attach ``outputs["loss_dict"]`` from
                :meth:`SeqVaeRawV4.compute_loss` (the raw single-phase objective).
            beta: KL weight for loss computation.
            lambda_full: Weight on the full raw-forecast NLL.
            lambda_base: Weight on the baseline raw-forecast NLL.

        Returns:
            The full 25-key forward-output dict from :meth:`SeqVaeRawV4.forward` (raw-shaped
            decoder tensors $(B, T, H, R)$), with an optional ``loss_dict`` key when
            ``compute_loss=True``.
        """
        fhr_raw, up_raw, mask = self._build_raw_input(batch)
        outputs = self.model(fhr_raw, up_raw, mask)

        if compute_loss:
            outputs["loss_dict"] = self.model.compute_loss(
                outputs,
                fhr_raw,
                mask,
                beta=beta,
                lambda_full=lambda_full,
                lambda_base=lambda_base,
            )

        return outputs
