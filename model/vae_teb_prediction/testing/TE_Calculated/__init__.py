"""Empirical Transfer Entropy vs VAE-KLD comparison pipeline.

Run as a module to execute the full comparison with defaults from the
``__main__`` block in ``te_kld_comparison``::

    python -m model.vae_teb_prediction.testing.TE_Calculated.te_kld_comparison

Or import :func:`run_comparison` to drive it programmatically.
"""

from model.vae_teb_prediction.testing.TE_Calculated.te_data_loader import (
    compute_data_quality_report,
    fuzzy_time_match,
    get_te_guids,
    load_te_data,
    normalize_guid,
    round_domain_start,
)
from model.vae_teb_prediction.testing.TE_Calculated.te_dtw import (
    dtw_align_per_guid,
    dtw_backend_name,
    paired_dataset_from_dtw,
)
from model.vae_teb_prediction.testing.TE_Calculated.te_kld_analysis import (
    CANDIDATE_KLD_COLS,
    CANDIDATE_TE_COLS,
    cluster_aware_bootstrap,
    compute_cross_guid_correlation,
    compute_per_guid_correlations,
    compute_pooled_correlation,
    concordance_analysis,
    correlation_matrix,
    export_summary,
    leave_one_guid_out_sensitivity,
    load_kld_from_inference,
    load_kld_from_metrics_csv,
    merge_te_kld,
    mutual_information_knn,
    per_dimension_kl_analysis,
    permutation_test_correlation,
    population_level_test,
    trend_agreement_analysis,
)
from model.vae_teb_prediction.testing.TE_Calculated.te_kld_comparison import (
    run_comparison,
)

__all__ = [
    "CANDIDATE_KLD_COLS",
    "CANDIDATE_TE_COLS",
    "cluster_aware_bootstrap",
    "compute_cross_guid_correlation",
    "compute_data_quality_report",
    "compute_per_guid_correlations",
    "compute_pooled_correlation",
    "concordance_analysis",
    "correlation_matrix",
    "dtw_align_per_guid",
    "dtw_backend_name",
    "export_summary",
    "fuzzy_time_match",
    "get_te_guids",
    "leave_one_guid_out_sensitivity",
    "load_kld_from_inference",
    "load_kld_from_metrics_csv",
    "load_te_data",
    "merge_te_kld",
    "mutual_information_knn",
    "normalize_guid",
    "paired_dataset_from_dtw",
    "per_dimension_kl_analysis",
    "permutation_test_correlation",
    "population_level_test",
    "round_domain_start",
    "run_comparison",
    "trend_agreement_analysis",
]
