"""Consecutive forecast tiling analysis.

Places anchors at every *h* steps (for a chosen horizon *h*) so that
forecast windows tile the signal continuously.  This reveals the model's
ability to reconstruct and extrapolate the signal as a contiguous forecast.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from model.transformer.tr_testing.base import TransformerTestRunner
from model.transformer.tr_testing.collectors import collect_consecutive_forecast


def run_consecutive_forecast_analysis(
    runner: TransformerTestRunner,
    class_loaders: Dict[str, Any],
    output_dir: Path,
    horizon: Optional[int] = None,
    n_samples: int = 10,
) -> Dict[str, Any]:
    """Run the consecutive forecast tiling analysis for all classes.

    For each class, collects ``n_samples`` segments, places anchors at
    stride = ``horizon`` to tile non-overlapping forecast windows, and
    generates per-sample diagnostic figures.

    Args:
        runner: TransformerTestRunner instance.
        class_loaders: Dict mapping class names to DataLoaders.
        output_dir: Output directory for plots.
        horizon: Forecast horizon for tiling (must be in
            ``config.horizons``).  Defaults to the maximum horizon.
        n_samples: Number of samples per class.

    Returns:
        Summary dict with figure paths per class.
    """
    from model.transformer.tr_testing.visualizers import (
        plot_consecutive_forecast,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if horizon is None:
        horizon = runner.config.max_horizon

    results: Dict[str, Any] = {"horizon": horizon}

    for class_name, loader in class_loaders.items():
        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"  Collecting {n_samples} consecutive forecast samples "
            f"for {class_name} (h={horizon})..."
        )
        samples = collect_consecutive_forecast(
            runner, loader, class_name,
            horizon=horizon, n_samples=n_samples,
        )

        paths: List[str] = []
        for sample in samples:
            guid = sample.get("guid", "unknown")
            epoch = sample.get("epoch", 0)
            base = f"{guid}_{int(epoch)}_consec_h{horizon}"
            fig_path = class_dir / f"{base}.pdf"

            try:
                plot_consecutive_forecast(sample, fig_path, runner.config)
                paths.append(str(fig_path))
            except Exception as e:
                logger.warning(
                    f"Consecutive forecast plot failed for "
                    f"{guid} epoch {epoch}: {e}"
                )

        results[class_name] = {
            "n_samples": len(samples),
            "n_plots": len(paths),
            "paths": paths,
        }
        logger.info(
            f"  Generated {len(paths)} consecutive forecast figures "
            f"for {class_name}"
        )

    return results
