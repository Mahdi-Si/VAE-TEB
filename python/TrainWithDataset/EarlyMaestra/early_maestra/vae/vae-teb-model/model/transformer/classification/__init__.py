"""Transformer-based classification pipeline for GUID-level sequence modeling.

Implements a time-aware GRU classifier that operates on segment-level embeddings
extracted from a frozen (or fine-tunable) ``CausalMultimodalTransformer``.  Each
20-minute segment is summarised as a 416-dim embedding comprising attention-pooled
or mean-pooled FHR and fused representations plus TE coupling statistics.  The
GRU processes these embeddings in chronological order with time-decay gating to
produce per-segment binary predictions (healthy vs unhealthy).

Modules:
    classification_model: Core ``TimeAwareGRUClassifier`` model.
    classification_trainer: Lightning wrapper + single-fold trainer.
    kfold_classification_trainer: K-fold parallel training orchestration.
    precompute_embeddings: Pre-compute transformer embeddings for fast training.
    evaluate_transformer_classifier: Inference, evaluation, cross-fold aggregation.
    evaluation_plots: Publication-quality evaluation plots.

Configuration:
    config_classification.yaml: Single source of truth for all pipeline settings.

Documentation:
    classification_model.md: Full architecture design specification.
    CLASSIFICATION_PIPELINE_REFERENCE.md: Complete LLM reference guide.
"""

__version__ = "1.0.0"
