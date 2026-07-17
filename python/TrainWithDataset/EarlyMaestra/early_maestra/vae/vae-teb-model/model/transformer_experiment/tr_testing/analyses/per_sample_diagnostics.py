"""Per-sample diagnostic figure generation (Category 1).

Produces extended multi-row diagnostic figures for selected samples,
matching the training callback style.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from model.transformer.tr_testing.base import TransformerTestRunner
from model.transformer.tr_testing.collectors import collect_full_sample_data


def run_per_sample_diagnostics(
    runner: TransformerTestRunner,
    class_loaders: Dict[str, Any],
    output_dir: Path,
    n_samples: int = 10,
) -> Dict[str, Any]:
    """Generate per-sample diagnostic figures for each class.

    Args:
        runner: TransformerTestRunner instance.
        class_loaders: Dict mapping class names to DataLoaders.
        output_dir: Output directory.
        n_samples: Number of samples per class.

    Returns:
        Summary dict with paths to generated figures.
    """
    from model.transformer.tr_testing.visualizers import (
        plot_sample_diagnostic,
        plot_sample_forecast_all_channels,
        plot_sample_latent_detail,
    )

    output_dir = Path(output_dir)
    results = {}

    for class_name, loader in class_loaders.items():
        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"  Collecting {n_samples} samples for {class_name}...")
        samples = collect_full_sample_data(
            runner, loader, class_name, n_samples=n_samples
        )

        paths = []
        for sample in samples:
            guid = sample.get("guid", "unknown")
            epoch = sample.get("epoch", 0)
            base = f"{guid}_{int(epoch)}"

            # 1. Main diagnostic figure (17-row overview)
            diag_path = class_dir / f"{base}.pdf"
            try:
                plot_sample_diagnostic(sample, diag_path, runner.config)
                paths.append(str(diag_path))
            except Exception as e:
                logger.warning(
                    f"Diagnostic plot failed for {guid} epoch {epoch}: {e}"
                )

            # 2. Full-channel forecast heatmaps (GT vs all heads)
            fc_path = class_dir / f"{base}_forecast_channels.pdf"
            try:
                plot_sample_forecast_all_channels(
                    sample, fc_path, runner.config
                )
                paths.append(str(fc_path))
            except Exception as e:
                logger.warning(
                    f"Forecast channels plot failed for {guid}: {e}"
                )

            # 3. Detailed latent visualization
            lat_path = class_dir / f"{base}_latent_detail.pdf"
            try:
                plot_sample_latent_detail(
                    sample, lat_path, runner.config
                )
                paths.append(str(lat_path))
            except Exception as e:
                logger.warning(
                    f"Latent detail plot failed for {guid}: {e}"
                )

        results[class_name] = {
            "n_samples": len(paths),
            "paths": paths,
        }
        logger.info(f"  Generated {len(paths)} diagnostic figures for {class_name}")

    return results
