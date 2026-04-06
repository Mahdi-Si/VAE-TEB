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
    from model.transformer.tr_testing.visualizers import plot_sample_diagnostic

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
            fname = f"{guid}_{int(epoch)}.pdf"
            out_path = class_dir / fname

            try:
                plot_sample_diagnostic(sample, out_path, runner.config)
                paths.append(str(out_path))
            except Exception as e:
                logger.warning(
                    f"Failed to plot sample {guid} epoch {epoch}: {e}"
                )

        results[class_name] = {
            "n_samples": len(paths),
            "paths": paths,
        }
        logger.info(f"  Generated {len(paths)} diagnostic figures for {class_name}")

    return results
