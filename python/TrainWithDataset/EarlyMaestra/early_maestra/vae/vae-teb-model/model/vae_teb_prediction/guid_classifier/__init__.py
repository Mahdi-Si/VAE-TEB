"""Temporal classification pipeline for GUID-level sequence modeling.

Implements a two-level architecture for fetal heart rate (FHR) classification:
a frozen VAE encoder produces per-segment latent representations, which are
then processed by a segment-level encoder and a temporal LSTM across the
segment sequence.  Each segment receives a context-informed binary prediction.

Modules:
    temporal_classification_model: Core ``TemporalVaeClassifier`` model.
    temporal_classifier_trainer: Lightning wrapper + single-fold trainer.
    kfold_temporal_trainer: K-fold parallel training orchestration.
    evaluate_temporal_classifier: Inference, evaluation, metrics, plots.
    length_bucket_sampler: Efficient batching of variable-length sequences.
    precompute_latents: Pre-compute VAE latents for fast training.
    benchmark_precompute: Timing/memory benchmark: on-the-fly vs pre-computed.

Configuration:
    config_temporal.yaml: Single source of truth for all pipeline settings.

Documentation:
    TEMPORAL_PIPELINE_PLAN.md: Full architecture design specification.
    TEMPORAL_CLASSIFIER_REFERENCE.md: Quick-start reference guide.
    SPRINT_PLAN.md: Development sprint plan and progress tracking.
    MEMORY_PROFILE.md: GPU memory profiling results and recommendations.
"""

__version__ = "1.0.0"
