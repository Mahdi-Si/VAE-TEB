r"""S4-T06: the shared eval runner for ``synthetic_v4`` (precedes the Sprint 5/6 split).

Builds a :class:`TestRunner` around a checkpoint's own model + a :class:`SyntheticRawDataModuleV4`
loader, so both the generic-eval branch (Sprint 5) and the ground-truth-grading branch (Sprint 6)
share one runnable entry point.

Unlike :meth:`TestRunner.from_checkpoint` -- which hard-asserts ``model_class == "SeqVaeRawV4"`` and
so cannot load the leaky negative control (stamped ``LeakyRawFrontendSeqVaeRawV4``) -- this runner
constructs the model itself from a small **synthetic class registry** and wraps a live
:class:`TestRunner` (the ``make_live_runner`` pattern), so it evaluates every arm including
``frontend_noncausal``. A checkpoint whose ``model_class`` is not in the registry raises before any
evaluation.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type

import torch
from torch.utils.data import DataLoader

from model.vae_teb_prediction.model.model_experiment.synthetic_v2.datamodule_v4 import (
    SyntheticRawDataModuleV4,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.leaky_frontend_v4 import (
    LeakyRawFrontendSeqVaeRawV4,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.reuse_v4 import (
    SeqVaeRawV4,
    TestRunner,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.run_pipeline_v4 import (
    StageContextV4,
    StageSpecV4,
    register_stage_v4,
)

logger = logging.getLogger(__name__)

#: The model classes a synthetic-v4 checkpoint may declare (prod vs the leaky negative control).
_SYNTH_MODEL_REGISTRY: Dict[str, Type[SeqVaeRawV4]] = {
    "SeqVaeRawV4": SeqVaeRawV4,
    "LeakyRawFrontendSeqVaeRawV4": LeakyRawFrontendSeqVaeRawV4,
}


def _load_synth_model(checkpoint_path: Path, *, device: torch.device) -> SeqVaeRawV4:
    r"""Reconstruct the checkpoint's model from its stamped ``model_class`` + ``model_kwargs``.

    Raises:
        ValueError: If the blob declares a ``model_class`` outside :data:`_SYNTH_MODEL_REGISTRY`
            (the model-class mismatch guard) or carries no ``model_kwargs``.
    """
    from train.graph_models_utils import load_checkpoint_strict

    blob = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    model_class = blob.get("model_class") if isinstance(blob, dict) else None
    if model_class not in _SYNTH_MODEL_REGISTRY:
        raise ValueError(
            f"checkpoint model_class={model_class!r} is not a synthetic_v4 model class; "
            f"expected one of {sorted(_SYNTH_MODEL_REGISTRY)}."
        )
    model_kwargs = blob.get("model_kwargs")
    if not model_kwargs:
        raise ValueError(
            f"checkpoint {checkpoint_path} has no 'model_kwargs'; cannot rebuild the model."
        )
    # logvar_clamp is stamped as a list; SeqVaeRawV4 expects a tuple.
    model_kwargs = dict(model_kwargs)
    if isinstance(model_kwargs.get("logvar_clamp"), list):
        lv = model_kwargs["logvar_clamp"]
        model_kwargs["logvar_clamp"] = (float(lv[0]), float(lv[1]))

    model = _SYNTH_MODEL_REGISTRY[model_class](**model_kwargs)
    if load_checkpoint_strict(model=model, checkpoint=blob) is None:
        raise RuntimeError(f"could not align checkpoint {checkpoint_path} into {model_class}.")
    return model.to(device).eval()


def _build_runner_and_loader_v4(
    checkpoint_path: Any,
    config: Dict[str, Any],
    *,
    benchmark: str = "G1_raw_v4",
    cache_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    batch_size: int = 2,
    split: str = "val",
    device: Optional[torch.device] = None,
) -> Tuple[TestRunner, DataLoader]:
    r"""Build a live :class:`TestRunner` + a synthetic loader from a checkpoint and config.

    Args:
        checkpoint_path: A synthetic-v4 checkpoint (stamped ``model_class`` + ``model_kwargs``).
        config: The (arm-resolved) config tree.
        benchmark: Active benchmark key under ``benchmarks``.
        cache_dir: Optional explicit cache dir (defaults to the config-resolved cache).
        output_dir: Runner output dir (defaults to a ``_eval`` sibling of the checkpoint).
        batch_size: Loader batch size.
        split: Which cached split to serve (``val`` by default, falling back to ``train``).
        device: Torch device (auto-detected when ``None``).

    Returns:
        ``(runner, loader)`` -- a :class:`TestRunner` around the checkpoint's model and a
        :class:`~torch.utils.data.DataLoader` over the synthetic cache.

    Raises:
        ValueError: If the checkpoint's ``model_class`` is not a synthetic-v4 class.
    """
    checkpoint_path = Path(checkpoint_path)
    dev = device if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    out_dir = Path(output_dir) if output_dir is not None else checkpoint_path.parent / "_eval"

    model = _load_synth_model(checkpoint_path, device=dev)
    geo = model.geometry
    runner = TestRunner(
        model=model,
        device=dev,
        output_dir=out_dir,
        warmup_steps=int(geo.warmup),
        horizon=int(geo.horizon),
        max_lag=int(getattr(model, "max_lag", 90)),
        use_up_st=bool(getattr(model, "use_up_st", True)),
    )

    # Eval is single-batch, single-process work; force num_workers=0 so the loader never spawns
    # DataLoader workers (fragile under Windows/pytest and unnecessary here).
    eval_config = copy.deepcopy(config)
    eval_config.setdefault("dataset_config", {}).setdefault("dataloader_config", {})[
        "num_workers"
    ] = 0
    dm = SyntheticRawDataModuleV4(eval_config, batch_size=batch_size, benchmark=benchmark,
                                  cache_dir=cache_dir)
    dm.setup("fit")
    # Route each split to its OWN loader with no cross-split fallback: an explicitly requested split
    # whose ``.npz`` is absent must fail loudly rather than be silently graded on another split's data
    # (the split fan-out asks for ``split in {train,val,test}``). The ``train`` split is graded through
    # the ordered, ``drop_last=False`` plain loader so its metrics drop no trailing samples and are
    # reproducible under the pilot cap -- ``train_dataloader()`` is ``shuffle=True``/``drop_last=True``
    # and is only for fitting. ``val`` is also the default for an unknown/None split.
    if split == "train":
        loader = dm.make_plain_train_loader()
    elif split == "test":
        loader = dm.test_dataloader()
    else:
        loader = dm.val_dataloader()
    if loader is None:
        raise FileNotFoundError(
            f"eval requested split {split!r} but that split's .npz is absent from the cache; "
            f"build it first (run the `build` stage)."
        )
    return runner, loader


# =============================================================================
# S5: generic raw eval (reused ``model_raw/testing`` kernels on the synthetic loader).
# =============================================================================
#: Per-analysis sample caps for the headline ``test_plots`` run and its pilot shrink.
_MAX_SAMPLES_FULL: Dict[str, int] = {
    "raw_forecast": 500, "calibration": 1000, "overlays": 10,
    "latent": 500, "kld_lag": 8, "attention": 200, "te_lag": 1000,
}
_MAX_SAMPLES_PILOT: Dict[str, int] = {
    "raw_forecast": 8, "calibration": 8, "overlays": 4,
    "latent": 8, "kld_lag": 4, "attention": 8, "te_lag": 8,
}


def _resolve_eval_checkpoint(ctx: StageContextV4) -> Path:
    r"""Locate the arm's trained checkpoint under ``results/<tag>/<arm>/``.

    Prefers ``final.ckpt``; falls back to ``best.ckpt``.

    Raises:
        FileNotFoundError: When neither checkpoint exists (train the arm first).
    """
    run_dir = ctx.run_dir()
    final_ckpt = run_dir / "final.ckpt"
    best_ckpt = run_dir / "best.ckpt"
    if final_ckpt.is_file():
        return final_ckpt
    if best_ckpt.is_file():
        return best_ckpt
    raise FileNotFoundError(
        f"no checkpoint for arm={ctx.arm!r} under {run_dir} (expected final.ckpt or best.ckpt); "
        f"run `--stage train --arm {ctx.arm}` first."
    )


def _eval_batch_size(config: Dict[str, Any], *, pilot: bool) -> int:
    r"""Resolve the eval batch size from ``general_config.batch_size`` (``test`` then ``train``)."""
    if pilot:
        return 2
    bs = config.get("general_config", {}).get("batch_size", {})
    return int(bs.get("test", bs.get("train", 8)))


def run_raw_metrics_v4(
    runner: TestRunner, loader: DataLoader, *, max_samples_forecast: int, max_samples_calib: int,
) -> Dict[str, Any]:
    r"""Run raw-forecast metrics (VAF/MSE/SNR/$R^2$ per horizon) + G10 calibration (S5-T01).

    Both reused ``model_raw/testing`` kernels write under ``runner.output_dir`` and never raise;
    their summary dicts are returned under ``raw_forecast`` / ``calibration``.
    """
    from model.vae_teb_prediction.model.model_raw.testing.analyses.calibration import (
        run_calibration_analysis,
    )
    from model.vae_teb_prediction.model.model_raw.testing.analyses.raw_forecast import (
        run_raw_forecast_analysis,
    )

    forecast = run_raw_forecast_analysis(runner, loader, max_samples=max_samples_forecast)
    calibration = run_calibration_analysis(runner, loader, max_samples=max_samples_calib)
    return {"raw_forecast": forecast, "calibration": calibration}


def run_overlays_v4(runner: TestRunner, loader: DataLoader, *, max_samples: int) -> Dict[str, Any]:
    r"""Emit qualitative raw forecast-vs-target overlays (denormalised bpm) (S5-T02).

    Reuses ``run_sample_diagnostics`` (which draws via ``visualizers_raw.plot_raw_forecast_overlay``)
    into ``<output_dir>/samples_diag/``.
    """
    from model.vae_teb_prediction.model.model_raw.testing.analyses.qualitative import (
        run_sample_diagnostics,
    )

    return run_sample_diagnostics(runner, loader, max_samples=max_samples)


def run_agnostic_analyses_v4(
    runner: TestRunner, loader: DataLoader, *, caps: Dict[str, int],
) -> Dict[str, Any]:
    r"""Run the domain-agnostic latent / KL-lag / attention / TE-lag analyses (S5-T03).

    Each analysis is model-agnostic (reads ``kld_per_t`` / ``te_lag_map`` / ``attn_weights`` from
    the forward dict) and is wrapped so one failing analysis never aborts the stage.
    """
    from model.vae_teb_prediction.model.model_raw.testing.analyses.attention_diagnostics import (
        run_attention_diagnostics,
    )
    from model.vae_teb_prediction.model.model_raw.testing.analyses.kld_lag_diagnostics import (
        run_kld_lag_diagnostics,
    )
    from model.vae_teb_prediction.model.model_raw.testing.analyses.latent import (
        run_latent_distribution_analysis,
    )
    from model.vae_teb_prediction.model.model_raw.testing.analyses.te_lag_analysis import (
        run_te_lag_class_analysis,
    )

    tasks = (
        ("latent", lambda: run_latent_distribution_analysis(
            runner, loader, max_samples=caps["latent"])),
        ("kld_lag", lambda: run_kld_lag_diagnostics(
            runner, loader, max_samples=caps["kld_lag"])),
        ("attention", lambda: run_attention_diagnostics(
            runner, loader, max_samples=caps["attention"])),
        ("te_lag", lambda: run_te_lag_class_analysis(
            runner, loader, max_samples=caps["te_lag"])),
    )
    out: Dict[str, Any] = {}
    for name, fn in tasks:
        try:
            out[name] = fn()
        except Exception as exc:  # noqa: BLE001 -- a failing agnostic analysis never gates the stage
            logger.warning("agnostic analysis %s failed: %s: %s", name, type(exc).__name__, exc)
            out[name] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def run_test_plots_v4(ctx: StageContextV4) -> int:
    r"""``test_plots`` stage: generic raw metrics + G10 calibration + overlays + agnostic analyses.

    Points the reused ``model_raw/testing`` kernels at the synthetic loader via the shared
    :func:`_build_runner_and_loader_v4`, routing every artefact under ``<run_dir>/test_plots/``.
    Model-generic only -- ground-truth grading is the separate ``eval`` stage (Sprint 6).
    """
    config = ctx.config
    caps = _MAX_SAMPLES_PILOT if ctx.pilot else _MAX_SAMPLES_FULL
    ckpt = _resolve_eval_checkpoint(ctx)
    runner, loader = _build_runner_and_loader_v4(
        ckpt, config, benchmark=ctx.benchmark,
        # Split-scoped (results/<tag>/<arm>/<split>/test_plots/) under split fan-out (S7-T03);
        # collapses to the arm root when split is None. Checkpoint stays at the arm root.
        output_dir=ctx.output_dir() / "test_plots",
        batch_size=_eval_batch_size(config, pilot=ctx.pilot),
        split=ctx.split or "val",
    )

    metrics = run_raw_metrics_v4(
        runner, loader, max_samples_forecast=caps["raw_forecast"],
        max_samples_calib=caps["calibration"],
    )
    run_overlays_v4(runner, loader, max_samples=caps["overlays"])
    run_agnostic_analyses_v4(runner, loader, caps=caps)

    print(f"[test_plots] arm={ctx.arm} -> {runner.output_dir}")
    rf = metrics.get("raw_forecast", {})
    if "raw_mse_mean" in rf:
        print(f"[test_plots] raw_mse_mean={rf['raw_mse_mean']:.4g} n={rf.get('n_samples')}")
    return 0


register_stage_v4(StageSpecV4(
    name="test_plots",
    run=run_test_plots_v4,
    order=45,
    model_dependent=True,
    fatal=True,
    help="generic raw-forecast metrics + G10 calibration + overlays + agnostic analyses (per arm)",
))
