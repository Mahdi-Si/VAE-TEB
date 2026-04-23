"""GUID-level stage-aware classifier on top of the pretrained VAE-TEB Lag-Attn v1.

Public entry points (imported lazily so the package is importable even when
heavy deps such as torch are unavailable at import time):

* :class:`GuidOutcomeClassifier` — top-level nn.Module that wraps the segment
  tokenizer, relative-time transformer, GUID head and auxiliary per-segment
  head.
* :class:`PlGuidClassifier` — PyTorch Lightning wrapper implementing the
  combined 3-class + binary + auxiliary loss.
* :func:`train_fold` — single-fold training entry point.
* :func:`run_kfold_parallel` — k-fold orchestrator that runs folds in parallel
  subprocesses with their own GPUs.
* :func:`evaluate_single_fold` — per-fold inference + three-metric-type
  evaluation pipeline.
* :func:`precompute_fold_latents` — build a frozen-VAE latent cache for one
  fold.

See ``PRD.md`` (next to this package) for the full design specification and
decision log.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "GuidOutcomeClassifier",
    "PlGuidClassifier",
    "train_fold",
    "run_kfold_parallel",
    "evaluate_single_fold",
    "precompute_fold_latents",
]

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from model.vae_teb_prediction.new_classifier.guid_cls_v1.evaluate_guid_classifier import (
        evaluate_single_fold,
    )
    from model.vae_teb_prediction.new_classifier.guid_cls_v1.guid_classifier import (
        GuidOutcomeClassifier,
    )
    from model.vae_teb_prediction.new_classifier.guid_cls_v1.kfold_trainer import (
        run_kfold_parallel,
    )
    from model.vae_teb_prediction.new_classifier.guid_cls_v1.lightning_module import (
        PlGuidClassifier,
    )
    from model.vae_teb_prediction.new_classifier.guid_cls_v1.precompute_latents import (
        precompute_fold_latents,
    )
    from model.vae_teb_prediction.new_classifier.guid_cls_v1.single_fold_trainer import (
        train_fold,
    )


def __getattr__(name: str):  # pragma: no cover - trivial lazy loader
    """Lazy attribute loader so importing the package is cheap."""
    if name == "GuidOutcomeClassifier":
        from model.vae_teb_prediction.new_classifier.guid_cls_v1.guid_classifier import (
            GuidOutcomeClassifier,
        )

        return GuidOutcomeClassifier
    if name == "PlGuidClassifier":
        from model.vae_teb_prediction.new_classifier.guid_cls_v1.lightning_module import (
            PlGuidClassifier,
        )

        return PlGuidClassifier
    if name == "train_fold":
        from model.vae_teb_prediction.new_classifier.guid_cls_v1.single_fold_trainer import (
            train_fold,
        )

        return train_fold
    if name == "run_kfold_parallel":
        from model.vae_teb_prediction.new_classifier.guid_cls_v1.kfold_trainer import (
            run_kfold_parallel,
        )

        return run_kfold_parallel
    if name == "evaluate_single_fold":
        from model.vae_teb_prediction.new_classifier.guid_cls_v1.evaluate_guid_classifier import (
            evaluate_single_fold,
        )

        return evaluate_single_fold
    if name == "precompute_fold_latents":
        from model.vae_teb_prediction.new_classifier.guid_cls_v1.precompute_latents import (
            precompute_fold_latents,
        )

        return precompute_fold_latents
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
