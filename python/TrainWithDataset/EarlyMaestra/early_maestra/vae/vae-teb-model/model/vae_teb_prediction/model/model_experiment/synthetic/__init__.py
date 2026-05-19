r"""Synthetic-data transfer-entropy (TE) validation suite for ``SeqVaeLagAttnV1``.

This package feeds the lag-attentive VAE-TEB model synthetic source/target
processes whose block transfer entropy $\mathrm{TE}^{(H)}_{U\to Y}$ is known
analytically, then measures whether the model's per-step KL surrogate
``kld_per_t`` recovers the *null*, *rank ordering*, *calibration*, and *lag
structure* of that ground-truth TE.

See ``model_experiment/model_validation.md`` for the benchmark theory and
``model_experiment/synthetic_te_validation_plan.md`` for the phased task
tracker.

Modules:
    analytic_te: Closed-form ground-truth TE formulas.
    generators: Synthetic data-generating processes (Benchmarks A / B / C / E,
        plus the reverse-roles directionality variant G).
    dataset: ``SyntheticTEDataset`` loading cached datasets.
    build_dataset: One-shot CLI -- generate, persist, and preview a dataset.
    visualize: Benchmark-aware plotting utilities for cached datasets.
    train_minimal: Standalone (non-Lightning) single-GPU training loop.
    gpu_pool: Task-parallel multi-GPU training scheduler (one model per GPU).
    evaluate_te: $\bar{K}$-vs-TE evaluation metrics (single / sweep / rho_null).
    lag_recovery: Attention lag-mass, input-level ablation and the two-lag
        (Benchmark E) two-band mass-ratio metrics.
    beta_sweep: Orchestration of the $\beta$ rate-distortion sweep.
    directionality: Forward-vs-reverse $\bar K$ directionality test (task 7.4).
    final_report: Collation of every phase's metrics into one report (task 7.6).
"""
