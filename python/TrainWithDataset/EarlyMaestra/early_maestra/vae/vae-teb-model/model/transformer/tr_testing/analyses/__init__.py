"""Analysis module orchestrator for the transformer testing pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from ..base import TransformerTestRunner


def run_all_analyses(
    runner: TransformerTestRunner,
    class_loaders: Dict[str, Any],
    guid_loaders: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Path] = None,
    max_samples: Optional[int] = None,
    skip_trajectory: bool = False,
    skip_per_sample: bool = False,
    n_diagnostic_samples: int = 10,
) -> Dict[str, Any]:
    """Run all analysis categories in sequence with error isolation.

    Args:
        runner: TransformerTestRunner instance.
        class_loaders: Dict mapping class names to DataLoaders.
        guid_loaders: Optional dict mapping class names to GUID-based DataLoaders.
        output_dir: Base output directory (defaults to runner.output_dir).
        max_samples: Maximum samples per class.
        skip_trajectory: Skip trajectory analysis.
        skip_per_sample: Skip per-sample diagnostics.
        n_diagnostic_samples: Samples per class for diagnostics.

    Returns:
        Summary dict with results from each analysis category.
    """
    if output_dir is None:
        output_dir = runner.output_dir

    results = {}
    class_names = list(class_loaders.keys())

    # 1. Dataset statistics (no model needed)
    try:
        from .dataset_stats import run_dataset_stats_analysis
        logger.info("Running dataset statistics analysis...")
        results["dataset_stats"] = run_dataset_stats_analysis(
            class_loaders, runner.ensure_dir("dataset_stats")
        )
    except Exception as e:
        logger.error(f"Dataset stats analysis failed: {e}")
        results["dataset_stats"] = {"error": str(e)}

    # 2. Forecasting analysis
    try:
        from .forecasting import run_forecasting_analysis
        logger.info("Running forecasting analysis...")
        results["forecasting"] = run_forecasting_analysis(
            runner, class_loaders, runner.ensure_dir("forecasting"),
            max_samples=max_samples,
        )
    except Exception as e:
        logger.error(f"Forecasting analysis failed: {e}")
        results["forecasting"] = {"error": str(e)}

    # 3. TE coupling analysis
    try:
        from .te_coupling import run_te_coupling_analysis
        logger.info("Running TE coupling analysis...")
        results["te_coupling"] = run_te_coupling_analysis(
            runner, class_loaders, runner.ensure_dir("te_coupling"),
            max_samples=max_samples,
        )
    except Exception as e:
        logger.error(f"TE coupling analysis failed: {e}")
        results["te_coupling"] = {"error": str(e)}

    # 4. Representation analysis
    try:
        from .representation import run_representation_analysis
        logger.info("Running representation analysis...")
        results["representation"] = run_representation_analysis(
            runner, class_loaders, runner.ensure_dir("representation"),
            max_samples=max_samples,
        )
    except Exception as e:
        logger.error(f"Representation analysis failed: {e}")
        results["representation"] = {"error": str(e)}

    # 5. Trajectory analysis
    if not skip_trajectory and guid_loaders:
        try:
            from .trajectory import run_trajectory_analysis
            logger.info("Running trajectory analysis...")
            results["trajectory"] = run_trajectory_analysis(
                runner, guid_loaders, runner.ensure_dir("trajectory"),
                max_samples=max_samples,
            )
        except Exception as e:
            logger.error(f"Trajectory analysis failed: {e}")
            results["trajectory"] = {"error": str(e)}

    # 6. Cross-class comparison
    try:
        from .cross_class import run_cross_class_analysis
        logger.info("Running cross-class analysis...")
        results["cross_class"] = run_cross_class_analysis(
            runner, class_loaders, runner.ensure_dir("cross_class"),
            max_samples=max_samples,
            forecast_results=results.get("forecasting"),
            te_results=results.get("te_coupling"),
        )
    except Exception as e:
        logger.error(f"Cross-class analysis failed: {e}")
        results["cross_class"] = {"error": str(e)}

    # 7. Per-sample diagnostics
    if not skip_per_sample:
        try:
            from .per_sample_diagnostics import run_per_sample_diagnostics
            logger.info("Running per-sample diagnostics...")
            results["per_sample"] = run_per_sample_diagnostics(
                runner, class_loaders,
                runner.ensure_dir("per_sample_diagnostics"),
                n_samples=n_diagnostic_samples,
            )
        except Exception as e:
            logger.error(f"Per-sample diagnostics failed: {e}")
            results["per_sample"] = {"error": str(e)}

    return results
